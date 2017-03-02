package sparkboost

import collection.mutable.Queue
import Double.MaxValue

import org.apache.spark.rdd.RDD

import sparkboost.utils.Comparison
import sparkboost.utils.Utils.safeLogRatio

object Learner extends Comparison {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("Learner")
    type RDDType = RDD[Instances]
    type BrAI = Broadcast[Array[Int]]
    type BrAD = Broadcast[Array[Double]]

    def findBestSplit(
            y: BrAI, w: BrAD, assign: Array[BrAI],
            nodes: Array[SplitterNode], maxDepth: Int,
            lossFunc: (Double, Double, Double, Double, Double) => Double
    )(data: Instances) = {
        val yLocal: Array[Int] = data.ptr.map(k => y.value(k))
        val wLocal: Array[Double] = data.ptr.map(k => w.value(k))
        val totalWeight = wLocal.sum
        val totalCount = wLocal.size

        def findBest(nodeIndex: Int, depth: Int) = {
            val localAssign = data.ptr.map(k => assign(nodeIndex).value(k))
            val alw = localAssign.zip(yLocal).zip(wLocal)
            val assignAndLabelsToWeights = alw.groupBy(_._1)

            val leftTotalPositiveWeight = assignAndLabelsToWeights((-1, 1)).map(_._2).sum
            val leftTotalPositiveCount = assignAndLabelsToWeights((-1, 1)).size
            val leftTotalNegativeWeight = assignAndLabelsToWeights((-1, -1)).map(_._2).sum
            val leftTOtalNegativeCount = assignAndLabelsToWeights((-1, -1)).size

            val rightTotalPositiveWeight = assignAndLabelsToWeights((1, 1)).map(_._2).sum
            val rightTotalPositiveCount = assignAndLabelsToWeights((1, 1)).size
            val rightTotalNegativeWeight = assignAndLabelsToWeights((1, -1)).map(_._2).sum
            val rightTotalNegativeCount = assignAndLabelsToWeights((1, -1)).size

            val rejectWeight = totalWeight -
                               (leftTotalNegativeWeight + leftTotalPositiveCount) -
                               (rightTotalNegativeWeight + rightTotalPositiveWeight)
            val rejectCount = totalCount -
                              (leftTOtalNegativeCount + leftTotalPositiveCount) -
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

            var minScore = Nil
            var splitVal = 0.0
            var onLeft = true
            var leftPredict = 0.0
            var rightPredict = 0.0

            val curIndex = 0
            for (((iloc, iy), iw) <- alw) {
                if (iloc < 0) {
                    // In left tree
                    if (compare(data.x(curIndex), leftLastSplitValue) > 0) {
                        val leftRemainPositiveWeight = leftTotalPositiveWeight - leftCurrPositiveWeight
                        val leftRemainNegativeWeight = leftTotalNegativeWeight - leftCurrNegativeWeight
                        val score = lossFunc(rejectWeight, leftCurrPositiveWeight, leftCurrNegativeWeight,
                                             leftRemainPositiveWeight, leftRemainNegativeWeight)
                        if (minScore == Nil || compare(score, minScore._1) < 0) {
                            val leftRemainPositiveCount = leftTotalPositiveCount - leftCurrPositiveCount
                            val leftRemainNegativeCount = leftTOtalNegativeCount - leftCurrNegativeCount
                            minScore = (
                                score,
                                (rejectWeight, leftCurrPositiveWeight, leftCurrNegativeWeight,
                                    leftRemainPositiveWeight, leftRemainNegativeWeight),
                                (rejectCount, leftCurrPositiveCount, leftCurrNegativeCount,
                                    leftRemainPositiveCount, leftRemainNegativeCount)
                            )
                            splitVal = lastSplitVal
                            onLeft = true
                            leftPredict = safeLogRatio(leftCurrPositiveWeight, leftCurrNegativeWeight)
                            rightPredict = safeLogRatio(leftRemainPositiveWeight, leftRemainNegativeWeight)
                        }
                        leftLastSplitIndex += 1
                        leftLastSplitValue = splits(leftLastSplitIndex)
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
                    if (compare(data.x(curIndex), rightLastSplitValue) > 0) {
                        val rightRemainPositiveWeight = rightTotalPositiveWeight - rightCurrPositiveWeight
                        val rightRemainNegativeWeight = rightTotalNegativeWeight - rightCurrNegativeWeight
                        val score = lossFunc(rejectWeight, rightCurrPositiveWeight, rightCurrNegativeWeight,
                                             rightRemainPositiveWeight, rightRemainNegativeWeight)
                        if (minScore == Nil || compare(score, minScore._1) < 0) {
                            val rightRemainPositiveCount = rightTotalPositiveCount - rightCurrPositiveCount
                            val rightRemainNegativeCount = rightTOtalNegativeCount - rightCurrNegativeCount
                            minScore = (
                                score,
                                (rejectWeight, rightCurrPositiveWeight, rightCurrNegativeWeight,
                                    rightRemainPositiveWeight, rightRemainNegativeWeight),
                                (rejectCount, rightCurrPositiveCount, rightCurrNegativeCount,
                                    rightRemainPositiveCount, rightRemainNegativeCount)
                            )
                            splitVal = lastSplitVal
                            onLeft = false
                            leftPredict = safeLogRatio(rightCurrPositiveWeight, rightCurrNegativeWeight)
                            rightPredict = safeLogRatio(rightRemainPositiveWeight, rightRemainNegativeWeight)
                        }
                        rightLastSplitIndex += 1
                        rightLastSplitValue = splits(rightLastSplitIndex)
                    }
                    if (iy > 0) {
                        rightCurrPositiveWeight += iw
                        rightCurrPositiveCount += 1
                    } else {
                        rightCurrNegativeWeight += iw
                        rightCurrNegativeCount += 1
                    }
                } // else if iloc is zero, ignore
                curIndex += 1
            }

            val curResult = (minScore, (nodeIndex, splitVal, data.index, splitVal,
                                        (leftPredict, rightPredict))
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

        findBest(Nil, 0, 0)
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
