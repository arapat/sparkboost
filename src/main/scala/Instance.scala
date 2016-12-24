package sparkboost

import scala.collection.immutable.Vector

class Instance(y: Int, x: Vector[Double], w: Double) {
    def apply(index: Int) = x(index)
}
