package sparkboost

class Instance(val y: Int, val X: Vector[Double], val w: Double,
               val scores: Vector[Double]) extends java.io.Serializable {
}

object Instance {
    def apply(y: Int, X: Vector[Double], w: Double = 1.0,
              scores: Vector[Double] = Vector[Double]()) = {
        new Instance(y, X, w, scores)
    }
}
