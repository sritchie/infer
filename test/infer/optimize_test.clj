(ns infer.optimize-test
  (:use clojure.test
        [infer.measures  :only [within]]
        infer.optimize))

(deftest make-step-fn-test
  ; f(x,y) = x^2 + 2*y, grad-f = (2x,2)
  (let [f (fn [[x y]] [(+ (* x x) (* 2 y)) [(* 2 x) 2]])        
        step-fn (#'infer.optimize/make-step-fn f [1 1] [1 0])]  
      (is (= (first (step-fn 1)) 6))
      (is (= (first (step-fn 0.5)) 4.25))))
         
(deftest backtracking-line-search-test
  ; f(x) = x^2 - 1
  (let [f (fn [[x]] [(- (* x x) 1) [(* 2 x)]])
        alpha (backtracking-line-search f [0] [1])
        step-fn (#'infer.optimize/make-step-fn f [0] [1])]
    (is (within 1.0e-4 (first (step-fn alpha)) -1))))
    
(deftest lbfgs-test
  ; f(x,y) = (x-2)^2 + (y+3)^2  
  ; min should be (2,-3)
  (let [f (fn [[x y]] 
            [(+ (* (- x 2) (- x 2)) (* (+ y 3) (+ y 3)))
             [(* 2 (- x 2)) (* 2 (+ y 3))]])
        [x-min y-min] (lbfgs-optimize f [0 0])]
    (is (within 1.0e-4 x-min 2))
    (is (within 1.0e-4 y-min -3))))    
