package sparkboost

import math.exp

import org.apache.spark.rdd.RDD

object UpdateFunc {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("UpdateFunc")
    type RDDType = RDD[Instances]

    def adaboostUpdateFunc(y: Int, w: Double, predict: Double) = w * exp(-y * predict)

    def logitboostUpdateFunc(y: Int, w: Double, predict: Double) = w / (1.0 + exp(y * predict))

    def updateFunc(data: RDDElementType, node: SplitterNode,
                   update: (Int, Double, Double) => Double): RDDElementType = {
        val leftPredict = node.leftPredict
        val rightPredict = node.rightPredict
        val f = node.check(_: Instance, false)
        def singleUpdate(inst: Instance) = {
            val c = f(inst)
            val pred = if (c > 0) leftPredict else if (c < 0) rightPredict else 0.0
            val updatedw = update(inst.y, inst.w, pred)
            Instance(inst.y, inst.X, updatedw, c +: inst.scores)
        }
        (data._1.map(singleUpdate), data._2, data._3)
    }

    def adaboostUpdate(train: RDDType, y: node: SplitterNode) = {
        rdd.map(updateFunc(_, node, adaboostUpdateFunc))
    }

    /*
    def logitboostUpdate(instances: RDDType, node: SplitterNode) = {
        def normalize(wsum: Double)(data: RDDElementType) = {
            (data._1.map(s => {
                s.setWeight(s.w / wsum)
                s
             }), data._2, data._3)
        }

        val raw = instances.map(updateFunc(_, node, logitboostUpdateFunc)).cache()
        val wsum = raw.map(_._1.map(_.w).reduce(_ + _)).reduce(_ + _)
        raw.map(normalize(wsum))
    }
    */
}
