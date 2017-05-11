package sparkboost

import math.exp
import math.max

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import sparkboost.utils.Comparison


// Update the weights after adding one more node into `nodes`
object UpdateFunc extends Comparison {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("UpdateFunc")
    def adaboostWeightUpdate(y: Int, w: Double, predict: Double) = w * exp(-y * predict)

    def logitboostWeightUpdate(y: Int, w: Double, predict: Double) = w / (1.0 + exp(y * predict))

    def update(glomTrain: Types.TrainRDDType, nodes: Types.ABrNode,
               updateRule: Types.WeightFunc): Types.TrainRDDType = {
        def isIn(x: Vector, idx: Int): Boolean = {
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

    def adaboostUpdate(glomTrain: Types.TrainRDDType, nodes: Types.ABrNode) = {
        update(glomTrain, nodes, adaboostWeightUpdate)
    }
}
