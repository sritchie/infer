(ns infer.spectral-test
  (:use clojure.test)
  (:use infer.matrix)
  (:use infer.measures)
  (:use infer.spectral))

(defn close-to
	([a b] (close-to a b 0.00001))
	([a b precision] (> precision (- a b))))

(deftest power-method
  (let [A (matrix [[2 1] [1 2]])
        B (matrix [[3 0] [0 0.5]])
        A-eigenset {:eigenvalue 1.5, :eigenvector (column-matrix [0.5 0.5])}
        B-eigenset {:eigenvalue 3, :eigenvector (column-matrix [1 0])}
        A-power (power-iter A)
        B-power (power-iter B)
        _ (print A-power)
        _ (print B-power)]
  (is (and 
        (close-to (:eigenvalue A-eigenset) (:eigenvalue A-power))
        (= (:eigenvector A-eigenset) (:eigenvector A-power))))
  (is (and 
        (close-to (:eigenvalue B-eigenset) (:eigenvalue B-power))
        (= (:eigenvector B-eigenset) (:eigenvector B-power))))))