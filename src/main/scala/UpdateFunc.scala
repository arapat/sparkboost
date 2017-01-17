package sparkboost

import math.exp

import org.apache.spark.rdd.RDD

object UpdateFunc {
    def adaboostUpdateFunc(y: Int, w: Double, pred: Double) = w * exp(-y * pred)

    def logitboostUpdateFunc(y: Int, w: Double, pred: Double) = w / (1.0 + exp(y * pred))

    def updateFunc(inst: Instance, node: SplitterNode, updateFunc: (Int, Double, Double) => Double) = {
        val c = node.check(inst.X, preChecked=false)
        val pred = if (c > 0) node.leftPredict else if (c < 0) node.rightPredict else 0.0
        val y = inst.y
        Instance(y, inst.X, updateFunc(y, inst.w, pred), inst.scores :+ c)
    }

    def adaboostUpdate(instances: RDD[Instance], node: SplitterNode) = {
        instances.map(updateFunc(_, node, adaboostUpdateFunc))
    }

    def logitboostUpdate(instances: RDD[Instance], node: SplitterNode) = {
        val raw = instances.map(updateFunc(_, node, logitboostUpdateFunc)).cache()
        val wsum = raw.map(_.w).reduce(_ + _)
        raw.map(t => Instance(t.y, t.X, t.w / wsum, t.scores))
    }
}
