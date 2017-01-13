package sparkboost

class Instance(val y: Int, val X: Vector[Double], val w: Double,
               val scores: Vector[Int]) extends java.io.Serializable {
    override def toString() = {
        s"Instance($y, X, $w, $scores)"
    }
}

object Instance {
    def apply(y: Int, X: Vector[Double], w: Double = 1.0,
              scores: Vector[Int] = Vector[Int]()) = {
        new Instance(y, X, w, scores)
    }
}
