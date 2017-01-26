package sparkboost

import collection.mutable.ListBuffer
import collection.mutable.Queue
import Double.MaxValue

import org.apache.spark.rdd.RDD

import utils.Comparison

object Learner extends Comparison {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("Learner")

    def findBestSplit(instances: List[Instance], index: Int, nodes: ListBuffer[SplitterNode], root: Int,
                      lossFunc: (Double, Double, Double, Double, Double) => Double) = {
        def search(curInsts: List[Instance], totWeight: Double) = {
            var minScore = MaxValue
            var splitVal = 0.0

            val posInsts = curInsts.filter(t => t.y > 0)
            val totPos = if (posInsts.size > 0) posInsts.map(_.w).reduce(_ + _) else 0.0
            val negInsts = curInsts.filter(t => t.y < 0)
            val totNeg = if (negInsts.size > 0) negInsts.map(_.w).reduce(_ + _) else 0.0
            val rej = totWeight - totPos - totNeg
            var leftPos = 0.0
            var leftNeg = 0.0
            var lastVal = if (curInsts.size > 0) curInsts(0).X(index) else 0.0
            for (t <- curInsts) {
                if (compare(t.X(index), lastVal) != 0) {
                    val score = lossFunc(rej, leftPos, leftNeg,
                                         totPos - leftPos, totNeg - leftNeg)
                    if (compare(score, minScore) < 0) {
                        minScore = score
                        splitVal = 0.5 * (t.X(index) + lastVal)
                    }
                }
                if (t.y > 0) {
                    leftPos += t.w
                } else {
                    leftNeg += t.w
                }
                lastVal = t.X(index)
            }
            (minScore, splitVal)
        }

        var minScore = MaxValue
        var bestNodeIndex = root
        var splitVal = 0.0
        var onLeft = false

        val totWeight = instances.map(_.w).reduce(_ + _)
        val queue = Queue[(Int, List[Instance])]()
        queue += ((root, instances))
        while (!queue.isEmpty) {
            val curObj = queue.dequeue()
            val nodeIndex = curObj._1
            val data = curObj._2

            // find a best split value on this node
            val leftInstances = data.filter {_.scores(nodeIndex) > 0}
            val leftRes = search(leftInstances, totWeight)
            val leftScore = leftRes._1
            val leftSplitVal = leftRes._2
            if (compare(leftScore, minScore) < 0) {
                minScore = leftScore
                bestNodeIndex = nodeIndex
                splitVal = leftSplitVal
                onLeft = true
            }

            val rightInstances = data.filter {_.scores(nodeIndex) < 0}
            val rightRes = search(rightInstances, totWeight)
            val rightScore = rightRes._1
            val rightSplitVal = rightRes._2
            if (compare(rightScore, minScore) < 0) {
                minScore = rightScore
                bestNodeIndex = nodeIndex
                splitVal = rightSplitVal
                onLeft = false
            }

            queue ++= nodes(nodeIndex).leftChild.map((_, leftInstances))
            queue ++= nodes(nodeIndex).rightChild.map((_, rightInstances))
        }
        (minScore, (bestNodeIndex, onLeft, ThresholdCondition(index, splitVal)))
    }

    def partitionedGreedySplit(
            instsGroup: RDD[(List[Instance], Int)], nodes: ListBuffer[SplitterNode],
            lossFunc: (Double, Double, Double, Double, Double) => Double,
            rootIndex: Int = 0) = {
        def callFindBestSplit(data: (List[Instance], Int)) = {
            findBestSplit(data._1, data._2, nodes, rootIndex, lossFunc)
        }

        val bestSplit = instsGroup.map(callFindBestSplit)
                                  .reduce {(a, b) => if (a._1 < b._1) a else b}
        println("Node " + nodes.size + " min score is " + bestSplit._1)
        bestSplit._2
    }

    def bulkGreedySplit(
            instsGroup: RDD[(List[Instance], Int)], nodes: ListBuffer[SplitterNode],
            lossFunc: (Double, Double, Double, Double, Double) => Double,
            rootIndex: Int = 0) = {
        val insts = instsGroup.map(_._1).reduce((a, b) => List.concat(a, b))
        val inst = insts.head
        val featureSize = inst.X.size
        val splits = (0 until featureSize) map (i => {
            findBestSplit(insts.sortWith(_.X(i) < _.X(i)).toList, i, nodes, rootIndex, lossFunc)
        })
        val bestSplit = splits.reduce {(a, b) => if (a._1 < b._1) a else b}
        println("Node " + nodes.size + " min score is " + bestSplit._1)
        bestSplit._2
    }
}
