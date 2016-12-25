package sparkboost

import sparkboost.utils.Comparison

trait Condition {
    def apply(instance: Instance) = check(instance)
    def check(instance: Instance): Boolean
}

class ThresholdCondition(val index: Int, val splitVal: Double) extends Condition with Comparison {
    var result: Option[Boolean] = None

    def check(instance: Instance) = {
        compare(instance(index), splitVal) <= 0
    }
}

object ThresholdCondition {
    def apply(index: Int, splitVal: Double) = new ThresholdCondition(index, splitVal)
}

class TrueCondition extends Condition {
    def check(instance: Instance) = true
}
