package sparkboost

import sparkboost.utils.Comparison

trait Condition extends java.io.Serializable {
    def apply(instance: Vector[Double]) = check(instance)
    def check(instance: Vector[Double]): Int
}

class ThresholdCondition(val index: Int, val splitVal: Double) extends Condition with Comparison {
    var result: Option[Boolean] = None

    def check(instance: Vector[Double]) = {
        if (compare(instance(index), splitVal) <= 0) {
            1
        } else {
            -1
        }
    }

    override def toString() = {
        "Index " + index + " <= " + splitVal
    }
}

object ThresholdCondition {
    def apply(index: Int, splitVal: Double) = new ThresholdCondition(index, splitVal)
}

class TrueCondition extends Condition {
    def check(instance: Vector[Double]) = 1
    override def toString() = "Always True"
}
