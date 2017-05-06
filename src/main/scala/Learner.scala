package sparkboost

import collection.mutable.ArrayBuffer
import scala.collection.mutable.{Map => MutableMap}
import Double.MaxValue
import math.min
import math.max

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector

import sparkboost.utils.Comparison
import sparkboost.utils.Utils.safeLogRatio

object Learner extends Comparison {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("Learner")
    type RDDType = RDD[Instances]
    type BrAI = Broadcast[Array[Int]]
    type BrAD = Broadcast[Array[Double]]
    type BrSV = Broadcast[SparseVector]
    type ABrNode = Array[Broadcast[SplitterNode]]
    type DoubleTuple2 = (Double, Double)
    type IntTuple2 = (Int, Int)
    type WeightsMap = collection.Map[Int, Map[Int, (DoubleTuple2, IntTuple2)]]

    // TODO: here I assumed batchId is always 0 so is ignored
    type ScoreType = (Double, BoardKey)
    type BoardKey = (Int, Int, Int, Boolean)
    type BoardType = Map[BoardKey, Double]
    type BrBoard = Broadcast[BoardType]

    type NodeInfoType = (Int, Int, Double, Boolean, Double)
    // type ResultType = (Double, NodeInfoType)  // (minScore, nodeInfo)
    type ResultType = (Double, Int, Int, Double, Boolean, Double)
    type TimerType = (Long, Long, Long, Long)
    val assignAndLabelsTemplate = Array(-1, 0, 1).flatMap(k => Array((k, 1), (k, -1))).map(_ -> (0.0, 0))

    def getOverallWeights(y: BrAI, w: BrAD, assign: Array[BrSV], nodes: ABrNode, maxDepth: Int,
                          range: Range)(data: Instances) = {
        def getWeights(node: SplitterNode): (Int, (DoubleTuple2, IntTuple2)) = {
            var totalPositiveWeight = 0.0
            var totalPositiveCount = 0
            var totalNegativeWeight = 0.0
            var totalNegativeCount = 0

            val curAssign = assign(node.index).value
            val DIM = curAssign.indices.size
            range.foreach(rawIdx => {
                val idx = rawIdx % DIM
                val ptr = curAssign.indices(idx)
                require(compare(curAssign.values(idx)) != 0)
                if (compare(curAssign.values(idx)) != 0) {
                    if (y.value(ptr) > 0) {
                        totalPositiveWeight += w.value(ptr)
                        totalPositiveCount += 1
                    } else {
                        totalNegativeWeight += w.value(ptr)
                        totalNegativeCount += 1
                    }
                }
            })

            (node.index, (
                (totalPositiveWeight, totalNegativeWeight),
                (totalPositiveCount, totalNegativeCount)))
        }

        (data.batchId, nodes.filter(_.value.depth < maxDepth)
                            .map(bcNode => getWeights(bcNode.value))
                            .toMap)
    }

    def findBestSplit(
            y: BrAI, w: BrAD, assign: Array[BrSV], nodeWeightsMap: Broadcast[WeightsMap],
            nodes: ABrNode, maxDepth: Int,
            logratio: Double, board: BrBoard, range: Range,
            thrA: Double, thrB: Double
    )(data: Instances): Iterator[(BoardKey, Double)] = {
        val timer = System.currentTimeMillis()

        def getSlot(x: Double) = {
            var left = -1
            var right = data.splits.size - 1
            while (left + 1 < right) {
                val mid = (left + right) / 2
                if (x <= data.splits(mid)) {
                    right = mid
                } else {
                    left = mid
                }
            }
            right
        }

        val nodeWeights = nodeWeightsMap.value(data.batchId)

        // time stamp
        val timeStamp1 = System.currentTimeMillis() - timer

        def findBest(node: SplitterNode): (List[(BoardKey, Double)], Boolean) = {
            var newScores = List[(BoardKey, Double)]()
            val timer = System.currentTimeMillis()

            val ((totalPositiveWeight, totalNegativeWeight),
                 (totalPositiveCount, totalNegativeCount)) = nodeWeights(node.index)

            // (1) fill the bins
            var weights = MutableMap[(Boolean, Int), Double]()
            var counts = MutableMap[(Boolean, Int), Int]()
            val curAssign = assign(node.index).value
            val DIM = curAssign.values.size
            assert(curAssign.values.size == curAssign.indices.size)
            range.map(rawIdx => {
                val idx = rawIdx % DIM
                val ptr = curAssign.indices(idx)
                val loc = curAssign.values(idx)
                require(compare(loc) != 0)
                if (compare(loc) != 0) {
                    val iy = y.value(ptr)
                    val slot = getSlot(data.x(ptr))
                    val key = (iy > 0, slot)
                    weights(key) = weights.getOrElse(key, 0.0) + w.value(ptr)
                    counts(key) = counts.getOrElse(key, 0) + 1
                }
            })

            // time stamp
            // val timeStamp1 = System.currentTimeMillis() - timer

            var positiveWeight = 0.0
            var positiveCount = 0
            var negativeWeight = 0.0
            var negativeCount = 0

            // (2) find a good splits using bins
            var i = 0
            var bingo = false
            while (i < data.splits.size - 1 && !bingo) {
            // for (i <- 0 until data.splits.size - 1) {
                positiveWeight += weights.getOrElse((true, i), 0.0)
                positiveCount += counts.getOrElse((true, i), 0)
                negativeWeight += weights.getOrElse((false, i), 0.0)
                negativeCount += counts.getOrElse((false, i), 0)

                // Check left tree
                // var totalRejectWeight = rejectWeight + totalPositiveWeight + totalNegativeWeight
                //                             - positiveWeight - negativeWeight
                // var totalRejectCount = rejectCount + totalPositiveCount + totalNegativeCount
                //                             - positiveCount - negativeCount
                val key1 = (node.index, data.index, i, true)
                val val1 = board.value.getOrElse(key1, 0.0) +
                            (positiveWeight * logratio - negativeWeight * logratio)
                newScores = (key1, val1) +: newScores

                // Check right tree
                // totalRejectWeight = rejectWeight + positiveWeight + negativeWeight
                // totalRejectCount = rejectCount + positiveCount + negativeCount
                val rPositiveWeight = totalPositiveWeight - positiveWeight
                val rNegativeWeight = totalNegativeWeight - negativeWeight
                val rPositiveCount = totalPositiveCount - positiveCount
                val rNegativeCount = totalNegativeCount - negativeCount
                val key2 = (node.index, data.index, i, false)
                val val2 = board.value.getOrElse(key2, 0.0) +
                            (rPositiveWeight * logratio - rNegativeWeight * logratio)
                newScores = (key2, val2) +: newScores

                bingo = (min(val1, val2) < thrA || thrB < max(val1, val2))
                i = i + 1
            }

            // time stamp
            // val timeStamp2 = System.currentTimeMillis() - timer - timeStamp1

            // ((minScore, (node.index, data.index, splitVal, splitEval, predict)),
            // ((minScore, node.index, data.index, splitVal, splitEval, predict),
            //     System.currentTimeMillis - timer)
            (newScores, bingo)
        }

        var res = List[(BoardKey, Double)]()
        var i = 0
        var bingo = false
        while (i < nodes.size && !bingo) {
            val node = nodes(i).value
            if (node.depth < maxDepth) {
                val t = findBest(node)
                res ++= t._1
                bingo = t._2
            }
            i += 1
        }
        res.toIterator

        // val timeStamp2 = System.currentTimeMillis() - timer - timeStamp1
        // val timeUnexplained = 0 // timeStamp2 - result.map(_._2).reduce(_ + _)
        // (result, (timeStamp1, timeStamp2, timeUnexplained, timer))

        // Will return following tuple:
        // (minScore, nodeInfo)
        // where minScore consists of
        //
        //     (score,
        //      (rej_weight, pos_weight, neg_weight),
        //      (rej_count,  pos_count,  neg_count)
        //     )
        //
        // and nodeInfo consists of
        //
        //     (bestNodeIndex, splitIndex, splitVal, splitEval, predict)
    }

    def partitionedGreedySplit(
            sc: SparkContext,
            train: RDDType, y: BrAI,
            w: BrAD, assign: Array[BrSV],
            nodes: ABrNode, maxDepth: Int,
            logratio: Double,
            board: BrBoard, range: Range,
            thrA: Double, thrB: Double): (BoardType, ScoreType, ScoreType) = {
        var tStart = System.currentTimeMillis()
        val nodeWeightsMap = train.filter(_.index == 0).map(
            getOverallWeights(y, w, assign, nodes, maxDepth, range)
        ).collectAsMap
        val bcWeightsMap = sc.broadcast(nodeWeightsMap)

        val timeWeightInfo = System.currentTimeMillis() - tStart
        tStart = System.currentTimeMillis()

        val f = findBestSplit(y, w, assign, bcWeightsMap, nodes, maxDepth,
                              logratio, board, range, thrA, thrB) _
        // suggests: List((minScore, nodeInfo, timer))
        val allSplits = train.filter(_.active)
                             .flatMap(f)
                             .cache()

        val timerMap = System.currentTimeMillis()
        allSplits.count
        val mapMs = System.currentTimeMillis - timerMap

        val scoreFirst = allSplits.map(t => (t._2, t._1)).cache()
        val maxScore = scoreFirst.max
        val minScore = scoreFirst.min
        scoreFirst.unpersist()
        val splitsMap = allSplits.collectAsMap
        allSplits.unpersist()

        println("FindWeakLearner took in total (ms) " + (System.currentTimeMillis() - tStart) +
                ", of which allSplits took (ms) " + mapMs)
        println

        // (minScore, node.index, data.index, splitVal, splitEval, predict)
        // suggests.map(t => (t._2, t._3, t._4, t._5, t._6)).toList
        (splitsMap.toMap, minScore, maxScore)
    }
}


/*
    def takeTopK(K: Int)(xs: List[ResultType], ys: List[ResultType]): List[ResultType] = {
        var xs1 = xs
        var ys1 = ys
        var ret = List[ResultType]()
        while ((K < 0 || ret.size < K) && xs1.size > 0 && ys1.size > 0) {
            (xs1, ys1) match {
                case (x :: xs2, y :: ys2) =>
                    if (x._1 < y._1) { ret = x +: ret; xs1 = xs2 }
                    else                   { ret = y +: ret; ys1 = ys2 }
            }
        }
        if (K < 0) {
            ret.reverse ++ xs1 ++ ys1
        } else {
            ret.reverse ++ xs1.take(K - ret.size) ++ ys1.take(K - ret.size)
        }
    }
*/
