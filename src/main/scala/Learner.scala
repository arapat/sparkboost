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
    type MinScoreType = (Double, (Double, Double, Double), (Int, Int, Int))
    type NodeInfoType = (Int, Int, Double, Boolean, Double)
    type ResultType = (MinScoreType, NodeInfoType, Array[Double])
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
    )(data: Instances) = {
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
                val minScore = (
                    score,
                    (rejectWeight, positiveWeight, negativeWeight),
                    (rejectCount, positiveCount, negativeCount)
                )
                val predict = 0.5 * safeLogRatio(positiveWeight, negativeWeight)
                Some((minScore, predict))
            } else {
                None
            }
        }

        val timer = System.nanoTime()
        val globalTimeLog = ArrayBuffer[Double]()
        val nodeWeights = nodeWeightsMap.value(data.batchId)

        def findBest(node: SplitterNode): ResultType = {
            val tstart = System.nanoTime()
            val timeLog = ArrayBuffer[Double]()
            var t0 = System.nanoTime()

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

            var positiveWeight = 0.0
            var positiveCount = 0
            var negativeWeight = 0.0
            var negativeCount = 0

            var minScore = (Double.MaxValue, (0.0, 0.0, 0.0), (0, 0, 0))
            var splitVal = 0.0
            var splitEval = true
            var predict = 0.0

            timeLog.append(System.nanoTime() - t0)
            t0 = System.nanoTime()
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
                    minScore._1,
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
                    minScore._1,
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
            timeLog.append(System.nanoTime() - t0)

            globalTimeLog.append(System.nanoTime() - tstart)
            (minScore, (node.index, data.index, splitVal, splitEval, predict), timeLog.toArray)
        }

        val result = nodes.filter(_.value.depth < maxDepth)
                          .map(node => findBest(node.value))
                          .reduce((a, b) => if (a._1._1 < b._1._1) a else b)
        val gtLog: Array[Double] = globalTimeLog.toArray
        (result._1, result._2,
            (System.nanoTime() - timer).toDouble +: ((result._3) ++ (Array(9999.0)) ++ gtLog))
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
            lossFunc: (Double, Double, Double) => Double) = {
        val SEC = 1000000
        var tStart = System.nanoTime()
        val totalWeight = w.value.sum
        val totalCount = w.value.size
        val nodeWeightsMap = train.filter(_.index == 0).map(
            getOverallWeights(y, w, assign, nodes, maxDepth, totalWeight, totalCount)
        ).collectAsMap
        val bcWeightsMap = sc.broadcast(nodeWeightsMap)

        val timeWeightInfo = (System.nanoTime() - tStart) / SEC
        tStart = System.nanoTime()

        val f = findBestSplit(y, w, assign, bcWeightsMap, nodes, maxDepth, lossFunc) _
        // suggests: List((minScore, nodeInfo, timer))
        val suggests = train.filter(_.active)
                            .map(f)
                            .sortBy(_._1._1)
                            .take(100)  // TODO: Make this a parameter
        println("Node " + nodes.size + " learner info")
        println("Collect weights info took (ms) " + timeWeightInfo)
        // println("Min score: " + "%.2f".format(minScore._1))
        // println("Reject weight/count: " + "%.2f".format(minScore._2._1) + " / " + minScore._3._1)
        // println("Pos weight/count: "  + "%.2f".format(minScore._2._2) + " / " + minScore._3._2)
        // println("Neg weight/count: "  + "%.2f".format(minScore._2._3) + " / " + minScore._3._3)

        println("FindWeakLearner took (ms) " + (System.nanoTime() - tStart) / SEC)
        // print("Timer details: ")
        // timer.foreach(k => print(k / SEC + ", "))
        println

        suggests.map(_._2).toList
    }
}
