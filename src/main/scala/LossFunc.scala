package sparkboost

import math.sqrt

object LossFunc {
    def lossfunc(rej: Double, leftPos: Double, leftNeg: Double,
                 rightPos: Double, rightNeg: Double) = {
        val vals = Array(rej, leftPos, leftNeg, rightPos, rightNeg)
        val sum = vals.reduce(_ + _)
        val Array(nRej, nLeftPos, nLeftNeg, nRightPos, nRightNeg) = vals.map(_ / sum)
        nRej + 2 * (sqrt(nLeftPos * nLeftNeg) + sqrt(nRightPos * nRightNeg))
    }
}
