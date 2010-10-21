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
    
    To serialize a classifier, use serialize-obj to get a data structure
    that can be serialized and load-classifier to return
    a ISparseClassifier protocol
            
    See test for example usage."
   :author "Aria Haghighi <me@aria42.com>"}
  (:use [infer.core :only [map-map, log-add]]
        [infer.measures :only [dot-product, sparse-dot-product]]
        [infer.optimize :only [remember-last, lbfgs-optimize]]        
        [clojure.contrib.map-utils :only [deep-merge-with]]))

;; ---------------------------------------------
;; Core abstraction
;; ---------------------------------------------

(defprotocol ISparseClassifier
 (labels [model] 
   "returns set of possible labels for task")
 (predict-label [model datum]
   "predict highest scoring label")
 (label-posteriors [model datum] 
   "return map of label -> posterior for datum, 
    which is a map from active feature to value")
  (serialize-obj [_]
    "A data structure that can be serialized and read"))
    
(declare read-classifier)    

;; ---------------------------------------------
;; Data indexing
;; ---------------------------------------------
                                      
(defn- index-data  
  "takes seq of [label datum] pairs and returns 
   sets of all labels and predicates (aka features)" 
  [labeled-data]
  (let [labels (into #{} (map first labeled-data))
        preds (loop [labeled-data labeled-data preds (transient #{})]
                (if (empty? labeled-data) (persistent! preds)
                    (recur (rest labeled-data) 
                           (reduce conj! preds (-> labeled-data first second keys)))))]
    [preds labels]))
                   

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

;; -----------------------------
;; Linear classifier
;; -----------------------------

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
  
(defrecord SparseLinearClassifier [weights-map]
 ISparseClassifier
 (labels [this] (keys weights-map))
 (predict-label [this datum]
   (apply max-key (make-score-fn datum) (labels this)))
 (label-posteriors [this datum]
   ((comp second (make-posterior-fn weights-map)) datum))
 (serialize-obj [this]
   {:data weights-map :type :linear}))  
   
;; ---------------------------------------------
;; Logistic Regression
;; ---------------------------------------------
           
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
    (loop [labeled-data labeled-data sum 0.0 grad {}]
      (if (empty? labeled-data) 
          [sum grad]
          (let [label (-> labeled-data first first)
                datum (-> labeled-data first second)
                [log-like local-grad] (log-reg-update labels post-fn label datum)]
            (recur (rest labeled-data)
                   (+ sum log-like)
                   (deep-merge-with + grad local-grad)))))))
                   
(defn- log-reg-parallel-compute
  [labels post-fn labeled-data]
  (let [obj-fn (make-log-reg-obj-fn labels post-fn)]
    (->> labeled-data
         (partition-all 200)         
         (pmap obj-fn)
         (reduce
           (fn [[sum-log-like sum-grad] [log-like grad]]
            [(+ sum-log-like log-like)
             (deep-merge-with + sum-grad grad)])
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
                      [log-like grad-map] 
                        (log-reg-parallel-compute labels 
                                                  (make-posterior-fn weight-map) 
                                                  (labeled-data-fn))
                      grad (decode-map-to-weights grad-map labels preds)
                      l2-penalty (/ (dot-product weights weights) (* 2 sigma-squared))
                      l2-grad (map #(/ % sigma-squared) weights)]
                  [(+ log-like l2-penalty) (map + grad l2-grad)]))                    
        weights (apply lbfgs-optimize (remember-last obj-fn) init-weights (-> opts seq flatten))
        weight-map (encode-weights-to-map weights labels preds)]
    (SparseLinearClassifier. weight-map)))    
 
;; ---------------------------------------------
;; Serialization
;; ---------------------------------------------
     
(defn load-classifier 
  "loads classifier from a clj data object that
  came from serialize-obj."
  [obj]
  (case (:type obj)
    :linear (SparseLinearClassifier. (:data obj))))