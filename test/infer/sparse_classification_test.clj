(ns infer.sparse-classification-test
  (:use clojure.test        
        infer.sparse-classification))
        
(deftest learn-logistic-regression-test
  (let [data [[:cat {:small 1 :fuzzy 1 :meows 1}]
              [:dog {:large 1 :fuzzy 1 :barks 1}]
              [:cat {:large 1 :meows 1}]]
        posterior-predict (learn-logistic-regression (fn [] data))]
    (is (> (-> {:small 1 :meows 1} posterior-predict :cat) 0.5))
        (> (-> {:fuzzy 1 :large 1} posterior-predict :dog) 0.5)))
    

        