package sparkboost

import sparkboost.utils.Comparison

trait Condition {
    def apply(instance: Instance) = check(instance)
    def check(instance: Instance): Boolean
}

class ThresholdCondition(index: Int, splitVal: Double) extends Condition with Comparison {
    var result: Option[Boolean] = None

    def check(instance: Instance) = {
        compare(instance(index), splitVal) <= 0
    }
}

class TrueCondition extends Condition {
    def check(instance: Instance) = true
}
