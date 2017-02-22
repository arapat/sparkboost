package sparkboost

import collection.mutable.Queue
import Double.MaxValue

import org.apache.spark.rdd.RDD

import utils.Comparison

object Learner extends Comparison {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("Learner")
    // RDDElement: Instances, feature index, split points
    type RDDElementType = (List[Instance], Int, List[Double])
    type RDDType = RDD[RDDElementType]

    def safeLogRatio(a: Double, b: Double) = {
        if (compare(a) == 0 && compare(b) == 0) {
            0.0
        } else {
            val ratio = math.min(10.0, math.max(a / b, 0.1))
            math.log(ratio)
        }
    }

    def findBestSplit(data: RDDElementType, nodes: Array[SplitterNode], root: Int,
                      lossFunc: (Double, Double, Double, Double, Double) => Double) = {
        def search(curInsts: List[Instance], index: Int, totWeight: Double, totInsts: Int, splits: List[Double]) = {
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
                            minScore = (score, rej, leftPos, leftNeg,
                                        totPos - leftPos, totNeg - leftNeg,
                                        rejCount, leftPosCount, leftNegCount,
                                        posCount - leftPosCount, negCount - leftNegCount)
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
        }

        val instances = data._1
        val index = data._2
        val splits = data._3

        var minScore = (MaxValue, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0)
        var bestNodeIndex = root
        var splitVal = 0.0
        var onLeft = false
        var preds = (0.0, 0.0)

        val totWeight = if (instances.size > 0) instances.map(_.w).reduce(_ + _) else 0.0
        val totExamples = instances.size
        val queue = Queue((root, instances, 0))
        while (!queue.isEmpty) {
            val curObj = queue.dequeue()
            val nodeIndex = curObj._1
            val data = curObj._2
            val depth = curObj._3

            // find a best split value on this node
            if (depth <= 0) {  // leftInstances.size >= (totExamples * 0.3).toInt) {
                val leftInstances = data.filter {_.scores(nodeIndex) > 0}
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
                // if (depth <= 0) {
                queue ++= nodes(nodeIndex).leftChild.map((_, leftInstances, depth + 1))
                // }

                val rightInstances = data.filter {_.scores(nodeIndex) < 0}
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
                // if (depth <= 0) {
                queue ++= nodes(nodeIndex).rightChild.map((_, rightInstances, depth + 1))
                // }
            }

        }
        (minScore, (bestNodeIndex, onLeft, index, splitVal, preds._1, preds._2))
    }

    def partitionedGreedySplit(
            instsGroup: RDDType, nodes: Array[SplitterNode],
            lossFunc: (Double, Double, Double, Double, Double) => Double,
            rootIndex: Int = 0) = {
        val bestSplit = instsGroup.map(findBestSplit(_, nodes, rootIndex, lossFunc))
                                  .reduce {(a, b) => if (a._1._1 < b._1._1) a else b}
        println("Node " + nodes.size + " min score")
        println("Min score: " + "%.2f".format(bestSplit._1._1))
        println("Rej weight/count: " + "%.2f".format(bestSplit._1._2) + " / " + bestSplit._1._7)
        println("Left pos weight/count: " + "%.2f".format(bestSplit._1._3) + " / " + bestSplit._1._8)
        println("Left neg weight/count: " + "%.2f".format(bestSplit._1._4) + " / " + bestSplit._1._9)
        println("Right pos weight/count: " + "%.2f".format(bestSplit._1._5) + " / " + bestSplit._1._10)
        println("Right neg weight/count: " + "%.2f".format(bestSplit._1._6) + " / " + bestSplit._1._11)
        bestSplit._2
    }

    /*
    def bulkGreedySplit(
            instsGroup: RDDType, nodes: Array[SplitterNode],
            lossFunc: (Double, Double, Double, Double, Double) => Double,
            rootIndex: Int = 0) = {
        val insts = instsGroup.map(_._1).reduce((a, b) => List.concat(a, b))
        val splits = instsGroup.map(_._2 -> _._3).toMap
        val inst = insts.head
        val featureSize = inst.X.size
        val splits = (0 until featureSize) map (i => {
            findBestSplit(insts.sortWith(_.X(i) < _.X(i)).toList, i, splits(i),
                          nodes, rootIndex, lossFunc)
        })
        val bestSplit = splits.reduce {(a, b) => if (a._1 < b._1) a else b}
        println("Node " + nodes.size + " min score is " + bestSplit._1)
        bestSplit._2
    }
    */
}
