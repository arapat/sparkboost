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
            val assignToWeights = localAssign.zip(wLocal).groupBy(_._1)

            var leftTotalPositiveWeight = assignToWeights(-1).map(_._2).sum
            var leftTotalPositiveCount = assignToWeights(-1).size

            var rightTotalPositiveWeight = assignToWeights(1).map(_._2).sum
            var rightTotalPositiveCount = assignToWeights(1).size

            var leftCurrPositiveWeight = 0.0
            var leftCurrPositiveCount = 0
            var leftTotalNegativeWeight = 0.0
            var leftCurrNegativeCount = 0

            var rightCurrPositiveWeight = 0.0
            var rightCurrPositiveCount = 0
            var rightTotalNegativeWeight = 0.0
            var rightCurrNegativeCount = 0

            var leftLastSplitIndex = 0
            var leftLastSplitValue = data.splits(leftLastSplitIndex)

            var rightLastSplitIndex = 0
            var rightLastSplitValue = data.splits(rightLastSplitIndex)

            val curIndex = 0
            for (iloc <- localAssign) {
                if (iloc < 0) {
                    fb
                } else if (iloc > 0) {

                }
                curIndex += 1
            }

            var minScore = (MaxValue, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0)
            var splitVal = 0.0

            val posInsts = curInsts.filter(t => t.y > 0)
            val posCount = posInsts.size
            val negInsts = curInsts.filter(t => t.y < 0)
            val negCount = negInsts.size
            if (posCount == 0 || negCount == 0) {
                (minScore, 0.0, (0.0, 0.0))
            } else {
                val totPos = posInsts.map(_.w).reduce(_ + _)
                val totNeg = negInsts.map(_.w).reduce(_ + _)
                val rej = totWeight - totPos - totNeg
                val rejCount = totInsts - posCount - negCount
                var leftPos = 0.0
                var leftPosCount = 0
                var leftNeg = 0.0
                var leftNegCount = 0
                var leftPred = 0.0
                var rightPred = 0.0
                var splitIndex = 0
                var lastSplitVal = splits(splitIndex)
                for (t <- curInsts) {
                    if (compare(t.X(index), lastSplitVal) > 0) {
                        val score = lossFunc(rej, leftPos, leftNeg,
                                             totPos - leftPos, totNeg - leftNeg)
                        if (compare(score, minScore._1) < 0) {
                            minScore =
                            leftPred = safeLogRatio(leftPos, leftNeg)
                            rightPred = safeLogRatio(totPos - leftPos, totNeg - leftNeg)
                            splitVal = lastSplitVal
                            /*
                            if (minScore < 1e-8) {
                                val rightPos = totPos - leftPos
                                val rightNeg = totNeg - leftNeg
                                val iter = nodes.size
                                val size = curInsts.size
                                log.info(s"debug: $iter, $size, $rej, $leftPos, $leftNeg, $rightPos, $rightNeg")
                            }
                            */
                        }
                        splitIndex += 1
                        lastSplitVal = splits(splitIndex)
                    }
                    if (t.y > 0) {
                        leftPos += t.w
                        leftPosCount += 1
                    } else {
                        leftNeg += t.w
                        leftNegCount += 1
                    }
                }
                (minScore, splitVal, (leftPred, rightPred))
            }

            // find a best split value on this node
            val leftInstances = data.filter {t => t.scores(t.scores.size - 1 - nodeIndex) > 0}
            val leftRes = search(leftInstances, index, totWeight, totExamples, splits)
            val leftScore = leftRes._1
            val leftSplitVal = leftRes._2
            if (compare(leftScore._1, minScore._1) < 0) {
                minScore = leftScore
                bestNodeIndex = nodeIndex
                splitVal = leftSplitVal
                preds = leftRes._3
                onLeft = true
            }

            val rightInstances = data.filter {t => t.scores(t.scores.size - 1 - nodeIndex) < 0}
            val rightRes = search(rightInstances, index, totWeight, totExamples, splits)
            val rightScore = rightRes._1
            val rightSplitVal = rightRes._2
            if (compare(rightScore._1, minScore._1) < 0) {
                minScore = rightScore
                bestNodeIndex = nodeIndex
                splitVal = rightSplitVal
                preds = rightRes._3
                onLeft = false
            }

            if (depth + 1 < maxDepth) {
                queue ++= nodes(nodeIndex).leftChild.map((_, leftInstances, depth + 1))
                queue ++= nodes(nodeIndex).rightChild.map((_, rightInstances, depth + 1))
            }
            (minScore, (bestNodeIndex, onLeft, index, splitVal, preds._1, preds._2))
        }

        findBest(Nil, 0, 0)
        // Will return following tuple:
        // (metaInfo, (bestNodeIndex, onLeft, splitIndex, splitVal, (leftPredict, rightPredict))
        // where metaInfo consists
        //
        //     (score,
        //      (rej_weight, leftPos_weight, leftNeg_weight, rightPos_weight, rightNeg_weight),
        //      (rej_count,  leftPos_count,  leftNeg_count,  rightPos_count,  rightNeg_count)
        //     )
    }

    def partitionedGreedySplit(
            train: RDDType, y: BrAI,
            w: BrAD, assign: Array[BrAI],
            nodes: Array[SplitterNode], maxDepth: Int,
            lossFunc: (Double, Double, Double, Double, Double) => Double) = {
        val (minScore, nodeInfo) = train.filter(_.active)
                                        .map(findBestSplit(y, w, assign, nodes, maxDepth, lossFunc))
                                        .reduce ((a, b) => {if (a._1._1 < b._1._1) a else b})
        println("Node " + nodes.size + " learner info")
        println("Min score: " + "%.2f".format(minScore._1))
        println("Reject weight/count: "    + "%.2f".format(minScore._2._1) + " / " + minScore._3._1)
        println("Left pos weight/count: "  + "%.2f".format(minScore._2._2) + " / " + minScore._3._2)
        println("Left neg weight/count: "  + "%.2f".format(minScore._2._3) + " / " + minScore._3._3)
        println("Right pos weight/count: " + "%.2f".format(minScore._2._4) + " / " + minScore._3._4)
        println("Right neg weight/count: " + "%.2f".format(minScore._2._5) + " / " + minScore._3._5)
        bestSplit._2
    }
}
