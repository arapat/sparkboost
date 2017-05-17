package sparkboost

import collection.mutable.ArrayBuffer

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector

object Types {
    type BaseInstance = (Int, SparseVector)
    type BaseRDD = RDD[BaseInstance]
    type BrAI = Broadcast[Array[Int]]
    type BrAD = Broadcast[Array[Double]]
    type BrSV = Broadcast[SparseVector]
    type BrNode = Broadcast[SplitterNode]

    type ABrNode = ArrayBuffer[Broadcast[SplitterNode]]
    // S_i, W_i(h), V_i(h), n
    type BoardInfo = (Double, Double, Double, Int)
    type BoardType = Map[Int, (Double, Array[BoardInfo])]
    type BoardElem = (Int, (Double, Array[BoardInfo]))
    type BoardList = List[BoardElem]

    // steps, gamma, val1, wsum1, wsq1, cnt1, wsum, nodeId, featureId, splitId, dir
    type ResultType = (Int, Double,
                       Double, Double, Double, Int,
                       Double, Int, Int, Int, Boolean)

    // glomId, instances, weights, board
    type GlomType = (Int, Array[BaseInstance], Array[Double], Double, BoardType)
    type GlomResultType = (Int, Array[BaseInstance], Array[Double], Double, BoardType, ResultType)
    type TrainRDDType = RDD[GlomType]
    type ResultRDDType = RDD[GlomResultType]


    type SampleFunc = Array[SplitterNode] => (RDD[BaseInstance], RDD[BaseInstance])
    type LossFunc = (Double, Double, Double) => Double
    type LearnerFunc = (SparkContext, TrainRDDType, ABrNode,
                        Int, Int,
                        Int, Int, Int, Double, Double) => (ResultRDDType, Double)
    type UpdateFunc = (TrainRDDType, ABrNode) => TrainRDDType
    type WeightFunc = (Int, Double, Double) => Double
}
