package sparkboost

import scala.collection.immutable.Vector

class Instance(val y: Int, val X: Vector[Double], var w: Double) {
    def apply(index: Int) = X(index)
}
