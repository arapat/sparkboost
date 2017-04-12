package sparkboost

import collection.mutable.ArrayBuffer
import scala.collection.mutable.{Map => MutableMap}
import Double.MaxValue

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
    type DoubleTuple3 = (Double, Double, Double)
    type IntTuple3 = (Int, Int, Int)
    type WeightsMap = collection.Map[Int, Map[Int, (DoubleTuple3, IntTuple3)]]
    type NodeInfoType = (Int, Int, Double, Boolean, Double)
    // type ResultType = (Double, NodeInfoType)  // (minScore, nodeInfo)
    type ResultType = (Double, Int, Int, Double, Boolean, Double)
    type TimerType = (Long, Long, Long, Long)
    val assignAndLabelsTemplate = Array(-1, 0, 1).flatMap(k => Array((k, 1), (k, -1))).map(_ -> (0.0, 0))

    def getOverallWeights(y: BrAI, w: BrAD, assign: Array[BrSV], nodes: ABrNode, maxDepth: Int,
                          totalWeight: Double, totalCount: Int)(data: Instances) = {
        def getWeights(node: SplitterNode): (Int, (DoubleTuple3, IntTuple3)) = {
            var totalPositiveWeight = 0.0
            var totalPositiveCount = 0
            var totalNegativeWeight = 0.0
            var totalNegativeCount = 0

            val curAssign = assign(node.index).value
            assert(curAssign.values.size == curAssign.indices.size)
            var idx = 0
            (0 until curAssign.values.size).foreach(idx => {
                val ptr = curAssign.indices(idx)
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

            val rejectWeight = totalWeight - (totalNegativeWeight + totalPositiveWeight)
            val rejectCount = totalCount - (totalNegativeCount + totalPositiveCount)

            (node.index, (
                (totalPositiveWeight, totalNegativeWeight, rejectWeight),
                (totalPositiveCount, totalNegativeCount, rejectCount)))
        }

        (data.batchId, nodes.filter(_.value.depth < maxDepth)
                            .map(bcNode => getWeights(bcNode.value))
                            .toMap)
    }

    def findBestSplit(
            y: BrAI, w: BrAD, assign: Array[BrSV], nodeWeightsMap: Broadcast[WeightsMap],
            nodes: ABrNode, maxDepth: Int,
            lossFunc: (Double, Double, Double) => Double
    )(data: Instances): (List[ResultType], TimerType) = {
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

        def updateMinScore(
            currBest: Double,
            rejectWeight: Double, positiveWeight: Double, negativeWeight: Double,
            rejectCount: Int, positiveCount: Int, negativeCount: Int
        ) = {
            var score = lossFunc(rejectWeight, positiveWeight, negativeWeight)
            if (compare(score, currBest) < 0) {
                val minScore = score
                val predict = 0.5 * safeLogRatio(positiveWeight, negativeWeight)
                Some((minScore, predict))
            } else {
                None
            }
        }

        val nodeWeights = nodeWeightsMap.value(data.batchId)

        // time stamp
        val timeStamp1 = System.currentTimeMillis() - timer

        def findBest(node: SplitterNode): (ResultType, Long) = {
            val timer = System.currentTimeMillis()

            val ((totalPositiveWeight, totalNegativeWeight, rejectWeight),
                 (totalPositiveCount, totalNegativeCount, rejectCount)) = nodeWeights(node.index)

            var weights = MutableMap[(Boolean, Int), Double]()
            var counts = MutableMap[(Boolean, Int), Int]()
            val curAssign = assign(node.index).value
            assert(curAssign.values.size == curAssign.indices.size)
            (0 until curAssign.values.size).map(idx => {
                val ptr = curAssign.indices(idx)
                val loc = curAssign.values(idx)
                if (compare(loc) != 0) {
                    val iy = y.value(ptr)
                    val slot = getSlot(data.xVec(ptr))
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

            var minScore = Double.MaxValue
            var splitVal = 0.0
            var splitEval = true
            var predict = 0.0

            for (i <- 0 until data.splits.size - 1) {
                positiveWeight += weights.getOrElse((true, i), 0.0)
                positiveCount += counts.getOrElse((true, i), 0)
                negativeWeight += weights.getOrElse((false, i), 0.0)
                negativeCount += counts.getOrElse((false, i), 0)

                // Check left tree
                var totalRejectWeight = rejectWeight + totalPositiveWeight + totalNegativeWeight
                                            - positiveWeight - negativeWeight
                var totalRejectCount = rejectCount + totalPositiveCount + totalNegativeCount
                                            - positiveCount - negativeCount
                updateMinScore(
                    minScore,
                    totalRejectWeight, positiveWeight, negativeWeight,
                    totalRejectCount, positiveCount, negativeCount
                ) match {
                    case Some((_minScore, _predict)) => {
                        minScore = _minScore
                        predict = _predict
                        splitVal = data.splits(i)
                        splitEval = true
                    }
                    case None => Nil
                }

                // Check right tree
                totalRejectWeight = rejectWeight + positiveWeight + negativeWeight
                totalRejectCount = rejectCount + positiveCount + negativeCount
                val rPositiveWeight = totalPositiveWeight - positiveWeight
                val rNegativeWeight = totalNegativeWeight - negativeWeight
                val rPositiveCount = totalPositiveCount - positiveCount
                val rNegativeCount = totalNegativeCount - negativeCount
                updateMinScore(
                    minScore,
                    totalRejectWeight, rPositiveWeight, rNegativeWeight,
                    totalRejectCount, rPositiveCount, rNegativeCount
                ) match {
                    case Some((_minScore, _predict)) => {
                        minScore = _minScore
                        predict = _predict
                        splitVal = data.splits(i)
                        splitEval = false
                    }
                    case None => Nil
                }
            }

            // time stamp
            // val timeStamp2 = System.currentTimeMillis() - timer - timeStamp1

            // ((minScore, (node.index, data.index, splitVal, splitEval, predict)),
            ((minScore, node.index, data.index, splitVal, splitEval, predict),
                System.currentTimeMillis - timer)
        }

        val result = nodes.filter(_.value.depth < maxDepth)
                          .map(node => findBest(node.value))
                          .sortBy(_._1._1)
                          .toList

        val timeStamp2 = System.currentTimeMillis() - timer - timeStamp1
        val timeUnexplained = timeStamp2 - result.map(_._2).reduce(_ + _)

        (result.map(_._1), (timeStamp1, timeStamp2, timeUnexplained, timer))
        // result.map(_._1)

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

    def partitionedGreedySplit(
            sc: SparkContext,
            train: RDDType, y: BrAI,
            w: BrAD, assign: Array[BrSV],
            nodes: ABrNode, maxDepth: Int,
            candidateSize: Int,
            lossFunc: (Double, Double, Double) => Double) = {
        var tStart = System.currentTimeMillis()
        val totalWeight = w.value.sum
        val totalCount = w.value.size
        val nodeWeightsMap = train.filter(_.index == 0).map(
            getOverallWeights(y, w, assign, nodes, maxDepth, totalWeight, totalCount)
        ).collectAsMap
        val bcWeightsMap = sc.broadcast(nodeWeightsMap)

        val timeWeightInfo = System.currentTimeMillis() - tStart
        tStart = System.currentTimeMillis()

        val f = findBestSplit(y, w, assign, bcWeightsMap, nodes, maxDepth, lossFunc) _
        // suggests: List((minScore, nodeInfo, timer))
        val allSplits = train.filter(_.active)
                             .map(f)
                             .cache()

        val timerMap = System.currentTimeMillis()
        allSplits.count
        println("allSplits map took (ms) " + (System.currentTimeMillis - timerMap))
        val slowest = allSplits.map(_._2).reduce((a, b) =>
            if ((a._1 + a._2 + a._4) > (b._1 + b._2 + b._4)) a else b
        )
        val (init, process, unexplained, startTime) = slowest
        println("Slowest worker: started " + (startTime - timerMap) + " ms later (" + init + ", " + process + "), unexplained " + unexplained)

        val timerReduce1 = System.currentTimeMillis
        val effectCandidateSize =
            if (candidateSize < 0) (allSplits.map(_._1.size).reduce(_ + _) * 0.1).ceil.toInt
            // if (candidateSize < 0) (allSplits.count * 0.1).ceil.toInt
            else                   candidateSize
        println("reduce1 takes (ms) " + (System.currentTimeMillis - timerReduce1))

        val timerReduce2 = System.currentTimeMillis
        val suggests = allSplits.map(_._1).reduce(takeTopK(effectCandidateSize))
        // val suggests = allSplits.glom().map(_.toList).reduce(takeTopK(effectCandidateSize))
        println("reduce2 takes (ms) " + (System.currentTimeMillis - timerReduce2))

        allSplits.unpersist()
        // println("Node " + nodes.size + " learner info")
        println("Number of candidates: " + suggests.size)
        println("Collect weights info took (ms) " + timeWeightInfo)

        println("FindWeakLearner took (ms) " + (System.currentTimeMillis() - tStart))
        // print("Timer details: ")
        // timer.foreach(k => print(k + ", "))
        println

        // suggests.map(_._2).toList
        suggests.map(t => (t._2, t._3, t._4, t._5, t._6)).toList
    }
}
