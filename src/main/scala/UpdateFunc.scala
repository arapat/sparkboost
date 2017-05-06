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

    def adaboostWeightUpdate(y: Int, w: Double, predict: Double) = w * exp(-y * predict)

    def logitboostWeightUpdate(y: Int, w: Double, predict: Double) = w / (1.0 + exp(y * predict))

    def update(train: RDDType, y: BrAI, fa: BrSV, w: BrAD, node: Broadcast[SplitterNode],
               updateFunc: (Int, Double, Double) => Double): (SparseVector, Array[Double]) = {
        val curIndex = node.value.splitIndex
        val pred = node.value.pred
        val results = train.filter(_.index == max(0, curIndex)).flatMap(insts =>
            (0 until fa.value.indices.size).map(idx => {
                val ptr = fa.value.indices(idx)
                val faPredict = fa.value.values(idx)
                val ix = insts.x(ptr)
                val iy = y.value(ptr)
                val iw = w.value(ptr)
                val assign = (compare(faPredict) != 0) && node.value.check(ix, curIndex, true)
                val predict = if (assign) pred else 0.0
                val nw = updateFunc(iy, iw, predict) - iw
                (ptr, (assign, nw))
            }).filter(_._2._1)
        ).collect()
        val (indices, values) = results.unzip
        val (assign, weights) = values.unzip
        val assignVec = new SparseVector(w.value.size, indices, assign.map(t => if (t) 1.0 else 0.0))
        val wDelta = new SparseVector(w.value.size, indices, weights)
        val wVec = (0 until w.value.size).map(idx => w.value(idx) + wDelta(idx)).toArray
        // val wsum = wVec.reduce(_ + _)
        // val wVecNorm = wVec.map(_ / wsum)
        (assignVec, wVec)
    }

    def adaboostUpdate(train: RDDType, y: BrAI, fa: BrSV, w: BrAD, node: Broadcast[SplitterNode]) = {
        update(train, y, fa, w, node, adaboostWeightUpdate)
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
