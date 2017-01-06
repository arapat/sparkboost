package sparkboost

import math.exp

import org.apache.spark.rdd.RDD

object UpdateFunc {
    type Instance = (Int, Vector[Double], Double)

    def adaboostUpdate(instances: RDD[Instance], node: SplitterNode) = {
        instances.map(
            t => (t._1, t._2, t._3 * exp(-t._1 * node.predict(t._2, preChecked=false)))
        )
    }

    def logitboostUpdate(instances: RDD[Instance], node: SplitterNode) = {
        instances.map {
            t => (t._1, t._2, t._3 / (1.0 + exp(t._1 * node.predict(t._2, preChecked=false))))
        }
    }
}
