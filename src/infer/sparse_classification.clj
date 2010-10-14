(ns infer.sparse-classification
  {:doc "Implements sparse classification algorithms. This ns in contrast
    to infer.{learning,linear-models} is geared towards classifying sparse feature
    vectors rather than dense matrix encoding. The dense encoding
    isn't currently great for large feature spaces (like you get in NLP problems).
    The classifiers here should work for low millions of features and lots of data
    (assuming each datum has few number of non-zero feature);
    uses infer.optimize to find parameters to maximize objectives, which 
    also scales well with lots of data. Objective function computation
    is parallelized. 
    
    Each datum in this ns is represented by a map from feature to value. For instance,
    for doing category labeling of a document you might have a datum like:
    {
      \"has-word='Obama'\": 1
      \"from-base-url='cnn.com'\": 1
      \"referring-tweet-hashtag='#politics'\": 1
    }    
    
    Currently implemented:
    - logistic regression (maximum entropy classification)
    - one-vs-all ranking 
            
    See test for example usage."
   :author "Aria Haghighi <me@aria42.com>"}
  (:use [infer.core :only [map-map, log-add]]
        [infer.measures :only [dot-product, sparse-dot-product]]
        [infer.optimize :only [remember-last, lbfgs-optimize]]        
        [clojure.contrib.map-utils :only [deep-merge-with]]))
            
(defn- index-data  
  "takes seq of [label datum] pairs and returns 
   sets of all labels and predicates (aka features)" 
  [labeled-data]
  (reduce
    (fn [[preds labels] [label datum]]
      [(reduce conj preds (keys datum))
       (conj labels label)])         
    [#{} #{}]
    labeled-data))
                   

(defn- encode-weights-to-map 
  "converts flat seq of weights (model params) into nested map label => pred => weight.
   Assumes weights have been partitioned by labels. Uses ordering from
   labels and preds to intrepet weights."
  [weights labels preds]
  (let [all-label-weights (partition-all (count preds) weights)]
    (->> (zipmap labels all-label-weights) ; label -> label weights
         (map-map                          ; label -> (pred -> weight) 
           (fn [label-weights]
             (zipmap preds label-weights))))))
          
(defn- decode-map-to-weights
  "converts nested map label => pred => weight to flat seq 
   of weights for optimization"
  [weight-map labels preds]
  (apply concat
    (for [label labels :let [label-weights (weight-map label)]]
      (map label-weights preds))))

(defn- make-score-fn 
  "returns fn: label -> unnormalized log-probability of label"
  [weights-map]
  (fn [datum]
    (map-map
      (fn [weights] (sparse-dot-product datum weights))        
      weights-map)))


(defn- make-posterior-fn
  "return fn: label -> [logZ posterior-map], where
   posterior-map label -> posterior"
  [weights-map]
  (let [score-fn (make-score-fn weights-map)]
    (fn [datum]
      (let [labels (keys weights-map)
            scores (score-fn datum)
            log-z (log-add (vals scores))]
        [log-z
         (map-map #(Math/exp (- % log-z)) scores)]))))
         
(defn- log-reg-update [labels post-fn gold-label datum]
  (let [post-res (post-fn datum)
        log-z (first post-res)
        posts (second post-res)
        empirical (reduce
                    (fn [res [k v]]
                      (assoc-in res [gold-label k] v))
                    {}
                    datum)
        expected (map-map
                    (fn [post]
                      (map-map 
                        (fn [v] (* v post))
                        datum))
                    posts)]
    [(- (Math/log (posts gold-label)))
     (deep-merge-with - expected empirical)]))         
         
(defn- make-log-reg-obj-fn 
  [labels post-fn]
  (fn [labeled-data]
    (->> labeled-data
         (map (fn [labeled-datum] 
                (log-reg-update labels post-fn 
                                (first labeled-datum) 
                                (second labeled-datum))))
         (reduce
           (fn [res elem]
             [(+ (first res) (first elem))
              (deep-merge-with + (second res) (second elem))])
           [0.0 {}]))))
  
(defn learn-logistic-regression
  "Returns fn: datum => posterior, where the posterior is a map
   from label to probability. You can get a classifier by composing
   the function with (fn [m] (first (apply max-key second m))).
   
   Each datum is simply a map from predicates (features) to value 
   (for binary features, this is just 1). All absent features assumed to be 0. 
   
   labeled-data-fn: callback to return 
   seq of [label datum] which is labeled data. 
   Seq of labels implicitly determined from labeled data. 
      
   Main option is sigma-squared (L2 regularization parameter). All options
   passed on to optimization (lbfgs-optimize in infer.optimize)
      
   TODO: Support L1 regularization and feature pruning"
  [labeled-data-fn &
   {:as opts
    :keys [sigma-squared, max-iters]
    :or {sigma-squared 1.0 max-iters 100}}]
  (let [[preds labels] (index-data (labeled-data-fn))
        init-weights (double-array (* (count preds) (count labels)))
        obj-fn (fn [weights]
                (let [weight-map (encode-weights-to-map weights labels preds)
                      post-fn (make-posterior-fn weight-map)
                      inner-obj-fn (make-log-reg-obj-fn labels post-fn)
                      [log-like grad-map] (inner-obj-fn (labeled-data-fn))
                      grad (decode-map-to-weights grad-map labels preds)
                      l2-penalty (/ (dot-product weights weights) (* 2 sigma-squared))
                      l2-grad (map #(/ % sigma-squared) weights)]
                  [(+ log-like l2-penalty) (map + grad l2-grad)]))                    
        weights (apply lbfgs-optimize (remember-last obj-fn) init-weights (-> opts seq flatten))
        weight-map (encode-weights-to-map weights labels preds)
        post-fn (comp second (make-posterior-fn weight-map))]
    post-fn))    