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
    type DoubleTuple6 = (Double, Double, Double, Double, Double, Double)
    type IntTuple6 = (Int, Int, Int, Int, Int, Int)
    type WeightsMap = collection.Map[Int, Map[Int, (DoubleTuple6, IntTuple6)]]
    type MinScoreType = (Double, (Double, Double, Double, Double, Double), (Int, Int, Int, Int, Int))
    type NodeInfoType = (Int, Boolean, Int, Double, (Double, Double))
    type ResultType = (MinScoreType, NodeInfoType, Array[Double])
    val assignAndLabelsTemplate = Array(-1, 0, 1).flatMap(k => Array((k, 1), (k, -1))).map(_ -> (0.0, 0))

    def getOverallWeights(y: BrAI, w: BrAD, assign: Array[BrSV], nodes: ABrNode, maxDepth: Int,
                          totalWeight: Double, totalCount: Int)(data: Instances) = {
        def getWeights(node: SplitterNode): (Int, (DoubleTuple6, IntTuple6)) = {
            var leftTotalPositiveWeight = 0.0
            var leftTotalPositiveCount = 0
            var leftTotalNegativeWeight = 0.0
            var leftTotalNegativeCount = 0

            var rightTotalPositiveWeight = 0.0
            var rightTotalPositiveCount = 0
            var rightTotalNegativeWeight = 0.0
            var rightTotalNegativeCount = 0

            val curAssign = assign(node.index).value
            assert(curAssign.values.size == curAssign.indices.size)
            var idx = 0
            while (idx < curAssign.values.size) {
                val ptr = curAssign.indices(idx)
                val (ia, iy, iw) = (curAssign.values(idx), y.value(ptr), w.value(ptr))
                if (ia < 0) {
                    if (iy > 0) {
                        leftTotalPositiveWeight += iw
                        leftTotalPositiveCount += 1
                    } else {
                        leftTotalNegativeWeight += iw
                        leftTotalNegativeCount += 1
                    }
                } else if (ia > 0) {
                    if (iy > 0) {
                        rightTotalPositiveWeight += iw
                        rightTotalPositiveCount += 1
                    } else {
                        rightTotalNegativeWeight += iw
                        rightTotalNegativeCount += 1
                    }
                }
                idx = idx + 1
            }

            val leftRejectWeight = totalWeight - (leftTotalNegativeWeight + leftTotalPositiveWeight)
            val leftRejectCount = totalCount - (leftTotalNegativeCount + leftTotalPositiveCount)
            val rightRejectWeight = totalWeight - (rightTotalNegativeWeight + rightTotalPositiveWeight)
            val rightRejectCount = totalCount - (rightTotalNegativeCount + rightTotalPositiveCount)

            (node.index,
                ((leftTotalPositiveWeight, leftTotalNegativeWeight, leftRejectWeight,
                    rightTotalPositiveWeight, rightTotalNegativeWeight, rightRejectWeight),
                (leftTotalPositiveCount, leftTotalNegativeCount, leftRejectCount,
                    rightTotalPositiveCount, rightTotalNegativeCount, rightRejectCount)))
        }

        (data.batchId, nodes.filter(_.value.depth < maxDepth)
                            .map(bcNode => getWeights(bcNode.value))
                            .toMap)
    }

    def findBestSplit(
            y: BrAI, w: BrAD, assign: Array[BrSV], nodeWeightsMap: Broadcast[WeightsMap],
            nodes: ABrNode, maxDepth: Int,
            lossFunc: (Double, Double, Double, Double, Double) => Double
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

        val timer = System.nanoTime()
        val globalTimeLog = ArrayBuffer[Double]()
        val nodeWeights = nodeWeightsMap.value(data.batchId)

        def findBest(node: SplitterNode): ResultType = {
            val tstart = System.nanoTime()
            val timeLog = ArrayBuffer[Double]()
            var t0 = System.nanoTime()

            val ((leftTotalPositiveWeight, leftTotalNegativeWeight, leftRejectWeight,
                    rightTotalPositiveWeight, rightTotalNegativeWeight, rightRejectWeight),
                 (leftTotalPositiveCount, leftTotalNegativeCount, leftRejectCount,
                    rightTotalPositiveCount, rightTotalNegativeCount, rightRejectCount)) = nodeWeights(node.index)

            var weights = MutableMap[(Boolean, Boolean, Int), Double]()
            var counts = MutableMap[(Boolean, Boolean, Int), Int]()
            val curAssign = assign(node.index).value
            assert(curAssign.values.size == curAssign.indices.size)
            var idx = 0
            while (idx < curAssign.values.size) {
                val ptr = curAssign.indices(idx)
                val loc = curAssign.values(idx)
                if (compare(loc) != 0) {
                    val iy = y.value(ptr)
                    val slot = getSlot(data.xVec(ptr))
                    val key = (loc > 0, iy > 0, slot)
                    weights(key) = weights.getOrElse(key, 0.0) + w.value(ptr)
                    counts(key) = counts.getOrElse(key, 0) + 1
                }
                idx = idx + 1
            }

            var leftCurrPositiveWeight = 0.0
            var leftCurrPositiveCount = 0
            var leftCurrNegativeWeight = 0.0
            var leftCurrNegativeCount = 0

            var rightCurrPositiveWeight = 0.0
            var rightCurrPositiveCount = 0
            var rightCurrNegativeWeight = 0.0
            var rightCurrNegativeCount = 0

            var minScore = (
                Double.MaxValue,
                (0.0, 0.0, 0.0, 0.0, 0.0),
                (0, 0, 0, 0, 0)
            )
            var splitVal = 0.0
            var onLeft = true
            var leftPredict = 0.0
            var rightPredict = 0.0

            timeLog.append(System.nanoTime() - t0)
            t0 = System.nanoTime()
            for (i <- 0 until data.splits.size - 1) {
                leftCurrPositiveWeight += weights((false, true, i))
                leftCurrPositiveCount += counts((false, true, i))
                leftCurrNegativeWeight += weights((false, false, i))
                leftCurrNegativeCount += counts((false, false, i))

                rightCurrPositiveWeight += weights((true, true, i))
                rightCurrPositiveCount += counts((true, true, i))
                rightCurrNegativeWeight += weights((true, false, i))
                rightCurrNegativeCount += counts((true, false, i))

                def updateMinScore(
                    currBest: Double,
                    rejectWeight: Double, currPositiveWeight: Double, currNegativeWeight: Double,
                    totalPositiveWeight: Double, totalNegativeWeight: Double,
                    rejectCount: Int, currPositiveCount: Int, currNegativeCount: Int,
                    totalPositiveCount: Int, totalNegativeCount: Int
                ) = {
                    val remainPositiveWeight = totalPositiveWeight - currPositiveWeight
                    val remainNegativeWeight = totalNegativeWeight - currNegativeWeight
                    var score = lossFunc(rejectWeight, currPositiveWeight, currNegativeWeight,
                                         remainPositiveWeight, remainNegativeWeight)
                    val remainPositiveCount = totalPositiveCount - currPositiveCount
                    val remainNegativeCount = totalNegativeCount - currNegativeCount
                    if (compare(score, currBest) < 0) {
                        val minScore = (
                            score,
                            (rejectWeight, currPositiveWeight, currNegativeWeight,
                                remainPositiveWeight, remainNegativeWeight),
                            (rejectCount, currPositiveCount, currNegativeCount,
                                remainPositiveCount, remainNegativeCount)
                        )
                        val leftPredict = safeLogRatio(currPositiveWeight, currNegativeWeight)
                        val rightPredict = safeLogRatio(remainPositiveWeight, remainNegativeWeight)
                        Some((minScore, true, leftPredict, rightPredict))
                    } else {
                        None
                    }
                }

                // Check left tree
                updateMinScore(
                    minScore._1, leftRejectWeight, leftCurrPositiveWeight, leftCurrNegativeWeight,
                    leftTotalPositiveWeight, leftTotalNegativeWeight,
                    leftRejectCount, leftCurrPositiveCount, leftCurrNegativeCount,
                    leftTotalPositiveCount, leftTotalNegativeCount
                ) match {
                    case Some((_minScore, _onLeft, _leftPredict, _rightPredict)) => {
                        minScore = _minScore
                        onLeft = _onLeft
                        leftPredict = _leftPredict
                        rightPredict = _rightPredict
                    }
                    case None => Nil
                }
                // Check right tree
                updateMinScore(
                    minScore._1, rightRejectWeight, rightCurrPositiveWeight, rightCurrNegativeWeight,
                    rightTotalPositiveWeight, rightTotalNegativeWeight,
                    rightRejectCount, rightCurrPositiveCount, rightCurrNegativeCount,
                    rightTotalPositiveCount, rightTotalNegativeCount
                ) match {
                    case Some((_minScore, _onLeft, _leftPredict, _rightPredict)) => {
                        minScore = _minScore
                        onLeft = _onLeft
                        leftPredict = _leftPredict
                        rightPredict = _rightPredict
                    }
                    case None => Nil
                }
            }
            timeLog.append(System.nanoTime() - t0)

            globalTimeLog.append(System.nanoTime() - tstart)
            (minScore, (node.index, onLeft, data.index, splitVal,
                (leftPredict, rightPredict)), timeLog.toArray)
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
        //      (rej_weight, leftPos_weight, leftNeg_weight, rightPos_weight, rightNeg_weight),
        //      (rej_count,  leftPos_count,  leftNeg_count,  rightPos_count,  rightNeg_count)
        //     )
        //
        // and nodeInfo consists of
        //
        //     (bestNodeIndex, onLeft, splitIndex, splitVal, (leftPredict, rightPredict)
    }

    def partitionedGreedySplit(
            sc: SparkContext,
            train: RDDType, y: BrAI,
            w: BrAD, assign: Array[BrSV],
            nodes: ABrNode, maxDepth: Int,
            lossFunc: (Double, Double, Double, Double, Double) => Double) = {
        val SEC = 1000000
        var tStart = System.nanoTime()
        val totalWeight = w.value.sum
        val totalCount = w.value.size
        val nodeWeightsMap = train.filter(_.index == 0).map(
            getOverallWeights(y, w, assign, nodes, maxDepth, totalWeight, totalCount)
        ).collectAsMap
        val bcWeightsMap = sc.broadcast(nodeWeightsMap)

        println("Collect weights info took (ms) " + (System.nanoTime() - tStart) / SEC)
        tStart = System.nanoTime()

        val f = findBestSplit(y, w, assign, bcWeightsMap, nodes, maxDepth, lossFunc) _
        val (minScore, nodeInfo, timer) = train.filter(_.active)
                                               .map(f)
                                               .reduce((a, b) => {if (a._1._1 < b._1._1) a else b})
        println("Node " + nodes.size + " learner info")
        println("Min score: " + "%.2f".format(minScore._1))
        println("Reject weight/count: "    + "%.2f".format(minScore._2._1) + " / " + minScore._3._1)
        println("Left pos weight/count: "  + "%.2f".format(minScore._2._2) + " / " + minScore._3._2)
        println("Left neg weight/count: "  + "%.2f".format(minScore._2._3) + " / " + minScore._3._3)
        println("Right pos weight/count: " + "%.2f".format(minScore._2._4) + " / " + minScore._3._4)
        println("Right neg weight/count: " + "%.2f".format(minScore._2._5) + " / " + minScore._3._5)

        println("FindWeakLearner took (ms) " + (System.nanoTime() - tStart) / SEC)
        print("Timer details: ")
        timer.foreach(k => print(k / SEC + ", "))
        println

        nodeInfo
    }
}
