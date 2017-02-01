package sparkboost

import collection.mutable.ListBuffer
import collection.mutable.Queue
import Double.MaxValue

import org.apache.spark.rdd.RDD

import utils.Comparison

object Learner extends Comparison {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("Learner")
    // RDDElement: Instances, feature index, split points
    type RDDElementType = (List[Instance], Int, List[Double])
    type RDDType = RDD[RDDElementType]

    def findBestSplit(data: RDDElementType, nodes: ListBuffer[SplitterNode], root: Int,
                      lossFunc: (Double, Double, Double, Double, Double) => Double) = {
        def search(curInsts: List[Instance], index: Int, totWeight: Double, splits: List[Double]) = {
            var minScore = MaxValue
            var splitVal = 0.0

            val posInsts = curInsts.filter(t => t.y > 0)
            val totPos = if (posInsts.size > 0) posInsts.map(_.w).reduce(_ + _) else 0.0
            val negInsts = curInsts.filter(t => t.y < 0)
            val totNeg = if (negInsts.size > 0) negInsts.map(_.w).reduce(_ + _) else 0.0
            val rej = totWeight - totPos - totNeg
            var leftPos = 0.0
            var leftNeg = 0.0
            var splitIndex = 1
            var lastSplitVal = splits(splitIndex)
            for (t <- curInsts) {
                if (compare(t.X(index), lastSplitVal) > 0) {
                    val score = lossFunc(rej, leftPos, leftNeg,
                                         totPos - leftPos, totNeg - leftNeg)
                    if (compare(score, minScore) < 0) {
                        minScore = score
                        splitVal = lastSplitVal
                    }
                }
                if (t.y > 0) {
                    leftPos += t.w
                } else {
                    leftNeg += t.w
                }
            }
            (minScore, splitVal)
        }

        val instances = data._1
        val index = data._2
        val splits = data._3

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
            val leftRes = search(leftInstances, index, totWeight, splits)
            val leftScore = leftRes._1
            val leftSplitVal = leftRes._2
            if (compare(leftScore, minScore) < 0) {
                minScore = leftScore
                bestNodeIndex = nodeIndex
                splitVal = leftSplitVal
                onLeft = true
            }

            val rightInstances = data.filter {_.scores(nodeIndex) < 0}
            val rightRes = search(rightInstances, index, totWeight, splits)
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
            instsGroup: RDDType, nodes: ListBuffer[SplitterNode],
            lossFunc: (Double, Double, Double, Double, Double) => Double,
            rootIndex: Int = 0) = {
        val bestSplit = instsGroup.map(findBestSplit(_, nodes, rootIndex, lossFunc))
                                  .reduce {(a, b) => if (a._1 < b._1) a else b}
        println("Node " + nodes.size + " min score is " + bestSplit._1)
        bestSplit._2
    }

    /*
    def bulkGreedySplit(
            instsGroup: RDDType, nodes: ListBuffer[SplitterNode],
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
