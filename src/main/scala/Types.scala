package sparkboost

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

    type ABrNode = Array[Broadcast[SplitterNode]]
    type BoardType = Map[Int, Array[Double]]
    type BoardElem = (Int, Array[Double])
    type BoardList = List[BoardElem]
    type BaseInstance = (Int, SparseVector)

    // steps, nodeId, featureId, splitId, dir
    type ResultType = (Int, Int, Int, Int, Boolean)

    // glomId, instances, weights, board
    type GlomType = (Int, Array[BaseInstance], Array[Double], BoardType)
    type GlomResultType = (Int, Array[BaseInstance], Array[Double], BoardType, ResultType)
    type TrainRDDType = RDD[GlomType]


    type SampleFunc = Array[SplitterNode] => (RDD[BaseInstance], RDD[BaseInstance])
    type BaseToCSCFunc = RDD[BaseInstance] => ColRDD
    type LossFunc = (Double, Double, Double) => Double
    type Suggest = (Int, Int, Double, Boolean, Double)
    type LearnerObj = (BoardType, ScoreType)
    type LearnerFunc = (SparkContext, ColRDD, BrAI, BrAD,
                        Array[BrSV], Array[BrNode], Int,
                        Double, BrBoard, Range, Double, Double) => LearnerObj
    type UpdateFunc = (ColRDD, BrAI, BrSV, BrAD, BrNode) => (SparseVector, Array[Double])
    type WeightFunc = (Int, Double, Double) => Double
}
