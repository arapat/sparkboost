package sparkboost.utils

trait Comparison {
    def compare(a: Double, b: Double, precision: Double = 1e-8) = {
        if (a - b < -precision) {
            -1
        } else if (a - b > precision) {
            1
        }
        0
    }
}
