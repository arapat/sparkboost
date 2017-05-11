package sparkboost

import math.exp
import math.max

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import sparkboost.Types
import sparkboost.utils.Comparison


// Update the weights after adding one more node into `nodes`
object UpdateFunc extends Comparison {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("UpdateFunc")
    def adaboostWeightUpdate(y: Int, w: Double, predict: Double) = w * exp(-y * predict)

    def logitboostWeightUpdate(y: Int, w: Double, predict: Double) = w / (1.0 + exp(y * predict))

    def update(glomTrain: Types.GlomType, nodes: ABrNode,
               updateRule: (Int, Double, Double) => Double): Types.GlomType = {
        def isIn(x: Vector, idx: Int) = {
            if (idx < 0) {
                true
            } else {
                val node = nodes(idx).value
                node.check(x(max(0, node.splitIndex))) && isIn(x, node.prtIndex)
            }
        }

        val nodeIdx = nodes.size - 1
        val pred = nodes.last.value.pred
        glomTrain.map(glom => {
            val (glomId, data, weights, board) = glom
            val weightItr = weights.iterator
            val newWeights = data.map {case (y, x) => {
                val w = weightItr.next
                if (isIn(x, nodeIdx)) {
                    updateRule(y, w, pred)
                } else {
                    w
                }
            }}
            (glomId, data, newWeights, board)
        }).cache()
    }

    def adaboostUpdate(glomTrain: Types.GlomType, nodes: ABrNode) = {
        update(glomTrain, nodes, adaboostWeightUpdate)
    }
}
