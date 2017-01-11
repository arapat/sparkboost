package sparkboost

import math.exp

import org.apache.spark.rdd.RDD

object UpdateFunc {
    def adaboostUpdateFunc(y: Int, w: Double, pred: Double) = w * exp(-y * pred)

    def logitboostUpdateFunc(y: Int, w: Double, pred: Double) = w / (1.0 + exp(y * pred))

    def updateFunc(inst: Instance, pred: Double, updateFunc: (Int, Double, Double) => Double) = {
        val y = inst.y
        Instance(y, inst.X, updateFunc(y, inst.w, pred), inst.scores :+ pred)
    }

    def adaboostUpdate(instances: RDD[Instance], node: SplitterNode) = {
        instances.map {t => (t, node.predict(t.X, preChecked=false))}
                 .map {tp => updateFunc(tp._1, tp._2, adaboostUpdateFunc)}
    }

    def logitboostUpdate(instances: RDD[Instance], node: SplitterNode) = {
        instances.map {t => (t, node.predict(t.X, preChecked=false))}
                 .map {tp => updateFunc(tp._1, tp._2, logitboostUpdateFunc)}
    }
}
