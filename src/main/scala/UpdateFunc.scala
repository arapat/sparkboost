package sparkboost

import math.exp

import org.apache.spark.rdd.RDD

object UpdateFunc {
    def adaboostUpdate(instances: RDD[Instance], node: SplitterNode) = {
        instances.foreach {
            t => t.w = t.w * exp(-t.y * node.predict(t, preChecked=false))
        }
    }

    def logitboostUpdate(instances: RDD[Instance], node: SplitterNode) = {
        instances.foreach {
            t => t.w = t.w / (1.0 + exp(t.y * node.predict(t, preChecked=false)))
        }
    }
}
