package sparkboost.utils

object Utils extends Comparison {
    // Limit the prediction score in a reasonable range
    def safeLogRatio(a: Double, b: Double) = {
        if (compare(a) == 0 && compare(b) == 0) {
            0.0
        } else {
            val ratio = math.min(10.0, math.max(a / b, 0.1))
            math.log(ratio)
        }
    }
}
