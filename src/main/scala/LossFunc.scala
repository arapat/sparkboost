package sparkboost

import math.sqrt

object LossFunc {
    def lossfunc(rej: Double, pos: Double, neg: Double) = {
        val vals = List(rej, pos, neg)
        val sum = vals.reduce(_ + _)
        val List(normalRej, normalPos, normalNeg) = vals.map(_ / sum)
        normalRej + 2 * sqrt(normalPos * normalNeg)
    }
}
