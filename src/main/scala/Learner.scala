package sparkboost

import collection.mutable.Map
import Double.MaxValue

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import sparkboost.utils.Comparison
import sparkboost.utils.Utils.safeLogRatio

object Learner extends Comparison {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("Learner")
    type RDDType = RDD[Instances]
    type BrAI = Broadcast[Array[Int]]
    type BrAD = Broadcast[Array[Double]]
    type MinScoreType = (Double, (Double, Double, Double, Double, Double), (Int, Int, Int, Int, Int))
    type NodeInfoType = (Int, Boolean, Int, Double, (Double, Double))
    type ResultType = (MinScoreType, NodeInfoType)
    val assignAndLabelsTemplate = Array(-1, 1).flatMap(k => Array((k, 1), (k, -1))).map(_ -> (0.0, 0))

    def findBestSplit(
            y: BrAI, w: BrAD, assign: Array[BrAI],
            nodes: Array[SplitterNode], maxDepth: Int,
            lossFunc: (Double, Double, Double, Double, Double) => Double
    )(data: Instances) = {
        val (yLocal, wLocal): (Array[Int], Array[Double]) =
            data.ptr.map(k => (y.value(k), w.value(k))).unzip
        val denseX: Array[Double] = data.x.toDense.values
        val totalWeight = wLocal.sum
        val totalCount = wLocal.size

        def findBest(nodeIndex: Int, depth: Int): ResultType = {
            val localAssign: Array[Int] = data.ptr.map(k => assign(nodeIndex).value(k))
            var idx = 0
            val assignAndLabelsToWeights = collection.mutable.Map(assignAndLabelsTemplate: _*)
            while (idx < localAssign.size) {
                val ay = (localAssign(idx), yLocal(idx))
                val (ws, wc) = assignAndLabelsToWeights(ay)
                assignAndLabelsToWeights(ay) = (ws + wLocal(idx), wc + 1)
                idx += 1
            }

            val (leftTotalPositiveWeight, leftTotalPositiveCount) = assignAndLabelsToWeights((-1, 1))
            val (leftTotalNegativeWeight, leftTotalNegativeCount) = assignAndLabelsToWeights((-1, -1))
            val (rightTotalPositiveWeight, rightTotalPositiveCount) = assignAndLabelsToWeights((1, 1))
            val (rightTotalNegativeWeight, rightTotalNegativeCount) = assignAndLabelsToWeights((1, -1))

            val rejectWeight = totalWeight -
                               (leftTotalNegativeWeight + leftTotalPositiveWeight) -
                               (rightTotalNegativeWeight + rightTotalPositiveWeight)
            val rejectCount = totalCount -
                              (leftTotalNegativeCount + leftTotalPositiveCount) -
                              (rightTotalNegativeCount + rightTotalPositiveCount)

            var leftCurrPositiveWeight = 0.0
            var leftCurrPositiveCount = 0
            var leftCurrNegativeWeight = 0.0
            var leftCurrNegativeCount = 0

            var rightCurrPositiveWeight = 0.0
            var rightCurrPositiveCount = 0
            var rightCurrNegativeWeight = 0.0
            var rightCurrNegativeCount = 0

            var leftLastSplitIndex = 0
            var leftLastSplitValue = data.splits(leftLastSplitIndex)

            var rightLastSplitIndex = 0
            var rightLastSplitValue = data.splits(rightLastSplitIndex)

            var minScore = (
                Double.MaxValue,
                (0.0, 0.0, 0.0, 0.0, 0.0),
                (0, 0, 0, 0, 0)
            )
            var splitVal = 0.0
            var onLeft = true
            var leftPredict = 0.0
            var rightPredict = 0.0

            idx = 0
            while (idx < denseX.size) {
                val ix = denseX(idx)
                val iloc = localAssign(idx)
                val iy = yLocal(idx)
                val iw = wLocal(idx)
                if (iloc < 0) {
                    // In left tree
                    if (compare(ix, leftLastSplitValue) > 0) {
                        val leftRemainPositiveWeight = leftTotalPositiveWeight - leftCurrPositiveWeight
                        val leftRemainNegativeWeight = leftTotalNegativeWeight - leftCurrNegativeWeight
                        val score = lossFunc(rejectWeight, leftCurrPositiveWeight, leftCurrNegativeWeight,
                                             leftRemainPositiveWeight, leftRemainNegativeWeight)
                        if (compare(score, minScore._1) < 0) {
                            val leftRemainPositiveCount = leftTotalPositiveCount - leftCurrPositiveCount
                            val leftRemainNegativeCount = leftTotalNegativeCount - leftCurrNegativeCount
                            minScore = (
                                score,
                                (rejectWeight, leftCurrPositiveWeight, leftCurrNegativeWeight,
                                    leftRemainPositiveWeight, leftRemainNegativeWeight),
                                (rejectCount, leftCurrPositiveCount, leftCurrNegativeCount,
                                    leftRemainPositiveCount, leftRemainNegativeCount)
                            )
                            splitVal = leftLastSplitValue
                            onLeft = true
                            leftPredict = safeLogRatio(leftCurrPositiveWeight, leftCurrNegativeWeight)
                            rightPredict = safeLogRatio(leftRemainPositiveWeight, leftRemainNegativeWeight)
                        }
                        leftLastSplitIndex += 1
                        leftLastSplitValue = data.splits(leftLastSplitIndex)
                    }
                    if (iy > 0) {
                        leftCurrPositiveWeight += iw
                        leftCurrPositiveCount += 1
                    } else {
                        leftCurrNegativeWeight += iw
                        leftCurrNegativeCount += 1
                    }
                } else if (iloc > 0) {
                    // In right tree
                    // In left tree
                    if (compare(ix, rightLastSplitValue) > 0) {
                        val rightRemainPositiveWeight = rightTotalPositiveWeight - rightCurrPositiveWeight
                        val rightRemainNegativeWeight = rightTotalNegativeWeight - rightCurrNegativeWeight
                        val score = lossFunc(rejectWeight, rightCurrPositiveWeight, rightCurrNegativeWeight,
                                             rightRemainPositiveWeight, rightRemainNegativeWeight)
                        if (compare(score, minScore._1) < 0) {
                            val rightRemainPositiveCount = rightTotalPositiveCount - rightCurrPositiveCount
                            val rightRemainNegativeCount = rightTotalNegativeCount - rightCurrNegativeCount
                            minScore = (
                                score,
                                (rejectWeight, rightCurrPositiveWeight, rightCurrNegativeWeight,
                                    rightRemainPositiveWeight, rightRemainNegativeWeight),
                                (rejectCount, rightCurrPositiveCount, rightCurrNegativeCount,
                                    rightRemainPositiveCount, rightRemainNegativeCount)
                            )
                            splitVal = rightLastSplitValue
                            onLeft = false
                            leftPredict = safeLogRatio(rightCurrPositiveWeight, rightCurrNegativeWeight)
                            rightPredict = safeLogRatio(rightRemainPositiveWeight, rightRemainNegativeWeight)
                        }
                        rightLastSplitIndex += 1
                        rightLastSplitValue = data.splits(rightLastSplitIndex)
                    }
                    if (iy > 0) {
                        rightCurrPositiveWeight += iw
                        rightCurrPositiveCount += 1
                    } else {
                        rightCurrNegativeWeight += iw
                        rightCurrNegativeCount += 1
                    }
                } // else if iloc is zero, ignore
                idx = idx + 1
            }

            val curResult = (minScore, (nodeIndex, onLeft, data.index, splitVal,
                                        (leftPredict, rightPredict)))
            if (depth + 1 < maxDepth) {
                val r1 = nodes(nodeIndex).leftChild.map(t => findBest(t, depth + 1))
                            .reduce((a, b) => {if (a._1._1 < b._1._1) a else b})
                val r2 = nodes(nodeIndex).rightChild.map(t => findBest(t, depth + 1))
                            .reduce((a, b) => {if (a._1._1 < b._1._1) a else b})
                val rChild = if (r1._1._1 < r2._1._1) r1 else r2
                if (curResult._1._1 < rChild._1._1) curResult else rChild
            } else {
                curResult
            }
        }

        findBest(0, 0)
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
            train: RDDType, y: BrAI,
            w: BrAD, assign: Array[BrAI],
            nodes: Array[SplitterNode], maxDepth: Int,
            lossFunc: (Double, Double, Double, Double, Double) => Double) = {
        val (minScore, nodeInfo) = train.filter(_.active)
                                        .map(findBestSplit(y, w, assign, nodes, maxDepth, lossFunc))
                                        .reduce((a, b) => {if (a._1._1 < b._1._1) a else b})

        println("Node " + nodes.size + " learner info")
        println("Min score: " + "%.2f".format(minScore._1))
        println("Reject weight/count: "    + "%.2f".format(minScore._2._1) + " / " + minScore._3._1)
        println("Left pos weight/count: "  + "%.2f".format(minScore._2._2) + " / " + minScore._3._2)
        println("Left neg weight/count: "  + "%.2f".format(minScore._2._3) + " / " + minScore._3._3)
        println("Right pos weight/count: " + "%.2f".format(minScore._2._4) + " / " + minScore._3._4)
        println("Right neg weight/count: " + "%.2f".format(minScore._2._5) + " / " + minScore._3._5)

        nodeInfo
    }
}
