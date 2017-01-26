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

    def adaboostUpdate(instances: RDD[(List[Instance], Int)], node: SplitterNode) = {
        instances.map(t => (t._1.map(updateFunc(_, node, adaboostUpdateFunc)).toList,
                            t._2))
    }

    def logitboostUpdate(instances: RDD[(List[Instance], Int)], node: SplitterNode) = {
        val raw = instances.map(
            t => (t._1.map(updateFunc(_, node, logitboostUpdateFunc)).toList,
                  t._2)
        ).cache()
        val wsum = raw.map(_._1.map(_.w).reduce(_ + _)).reduce(_ + _)
        raw.map(
            t => (t._1.map(s => Instance(s.y, s.X, s.w / wsum, s.scores)).toList,
                  t._2)
        )
    }
}
