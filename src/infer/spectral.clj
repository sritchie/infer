(ns infer.spectral
  "Fundamental spectral machine learning methods.  Power iteration.
  Spectral clustering.  Spectral ranking."
  (:use infer.matrix)
  (:use infer.measures)
  (:use infer.learning))

(defn power-iter
  "Computes the eigenvector associated with the largest eigenvalue.
  Returns a map with keys :eigenvalue, :eigenvector.  The eigenvector
  Is normalized to have length 1."
  ([A] (power-iter A (column-matrix (repeat (row-count A) 1))))
  ([A r] (power-iter A r 0.0000001))
  ([A r precision] (power-iter A r precision l1-norm))
	([A r precision norm-fcn]
  (let [r* (times A r)
        next-r (divide r* (norm-fcn r*))
        next-lambda (get-at (times (trans next-r) A next-r) 0 0)]
          (if (euclidean-convergence? precision (from-column-matrix r) (from-column-matrix next-r))
            {:eigenvalue next-lambda, :eigenvector next-r}
          (recur A next-r precision l1-norm)))))
