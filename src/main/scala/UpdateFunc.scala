package sparkboost

import math.exp
import math.max

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import sparkboost.utils.Comparison

object UpdateFunc extends Comparison {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("UpdateFunc")
    type RDDType = RDD[Instances]
    type BrAI = Broadcast[Array[Int]]
    type BrAD = Broadcast[Array[Double]]
    type BrSV = Broadcast[SparseVector]

    def adaboostUpdateFunc(y: Int, w: Double, predict: Double) = w * exp(-y * predict)

    def logitboostUpdateFunc(y: Int, w: Double, predict: Double) = w / (1.0 + exp(y * predict))

    def update(train: RDDType, y: BrAI, fa: BrSV, w: BrAD, node: Broadcast[SplitterNode],
               updateFunc: (Int, Double, Double) => Double): (SparseVector, Array[Double]) = {
        val curIndex = node.value.splitIndex
        val pred = node.value.pred
        val results = train.filter(_.index == max(0, curIndex)).flatMap(insts =>
            insts.x.toDense.values.zip(insts.ptr).map { case (ix, ipt) => {
                val iy = y.value(ipt)
                val iw = w.value(ipt)
                val faPredict = fa.value(ipt)
                val assign =
                    if (compare(faPredict) == 0.0) {
                        false
                    } else {
                        node.value.check(ix, curIndex, true)
                    }
                val predict = if (assign) pred else 0.0
                val nw = updateFunc(iy, iw, predict)
                (ipt, (assign, nw))
            }}
        ).sortByKey().map(_._2).collect()
        val (assignDense, nw) = results.unzip
        (new DenseVector(assignDense.map(t => if (t) 1.0 else 0.0)).toSparse, nw)
    }

    def adaboostUpdate(train: RDDType, y: BrAI, fa: BrSV, w: BrAD, node: Broadcast[SplitterNode]) = {
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
