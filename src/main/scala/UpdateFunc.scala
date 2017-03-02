package sparkboost

import math.exp
import math.max

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

object UpdateFunc {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("UpdateFunc")
    type RDDType = RDD[Instances]
    type BrAI = Broadcast[Array[Int]]
    type BrAD = Broadcast[Array[Double]]

    def adaboostUpdateFunc(y: Int, w: Double, predict: Double) = w * exp(-y * predict)

    def logitboostUpdateFunc(y: Int, w: Double, predict: Double) = w / (1.0 + exp(y * predict))

    def update(train: RDDType, y: BrAI, fa: BrAI, w: BrAD, node: SplitterNode,
               updateFunc: (Int, Double, Double) => Double): (Array[Int], Array[Double]) = {
        val curIndex = node.splitIndex
        val curOnLeft = node.onLeft
        val leftPredict = node.leftPredict
        val rightPredict = node.rightPredict
        val results = train.filter(_.index == max(0, curIndex)).flatMap(insts =>
            insts.x.toDense.values.zip(insts.ptr).map { case (ix, ipt) => {
                val iy = y.value(ipt)
                val iw = w.value(ipt)
                val faPredict = fa.value(ipt)
                val assign =
                    if (faPredict == 0 || faPredict < 0 && !curOnLeft || faPredict > 0 && curOnLeft) {
                        0
                    } else {
                        node.check(ix, curIndex, true)
                    }
                val predict = if (assign < 0) leftPredict else if (assign > 0) rightPredict else 0.0
                val nw = updateFunc(iy, iw, predict)
                (ipt, (assign, nw))
            }}
        ).sortByKey().map(_._2).collect()
        (results.map(_._1).toArray, results.map(_._2).toArray)
    }

    def adaboostUpdate(train: RDDType, y: BrAI, fa: BrAI, w: BrAD, node: SplitterNode) = {
        update(train, y, fa, w, node, adaboostUpdateFunc)
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
