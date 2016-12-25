package sparkboost

import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Queue
import Double.MaxValue

import org.apache.spark.rdd.RDD

import utils.Comparison

object Learner extends Comparison {
    def findBestSplit(instances: List[Instance], index: Int, nodes: ListBuffer[SplitterNode], root: Int,
                      lossFunc: (Double, Double, Double, Double, Double) => Double) {
        def search(curInsts: List[Instance], totWeight: Double) = {
            var minScore = MaxValue
            var splitVal = 0.0

            val totPos = curInsts.filter(t => compare(t.y, 0.0) > 0).map(_.w).reduce(_ + _)
            val totNeg = curInsts.filter(t => compare(t.y, 0.0) < 0).map(_.w).reduce(_ + _)
            val rej = totWeight - totPos - totNeg
            var leftPos = 0.0
            var leftNeg = 0.0
            for (i <- 0 until (curInsts.size - 1)) {
                if (curInsts(i).y > 0) {
                    leftPos += curInsts(i).w
                } else {
                    leftNeg += curInsts(i).w
                }
                if (compare(curInsts(i).X(index), curInsts(i + 1).X(index)) != 0) {
                    val rightPos = totPos - leftPos
                    val rightNeg = totNeg - leftNeg
                    val score = lossFunc(rej, leftPos, leftNeg, rightPos, rightNeg)
                    if (compare(score, minScore) < 0) {
                        minScore = score
                        splitVal = 0.5 * (curInsts(i).X(index) + curInsts(i + 1).X(index))
                    }
                }
            }
            (minScore, splitVal)
        }

        var minScore = MaxValue
        var bestNode = nodes(root)
        var splitVal = 0.0
        var onLeft = false

        val totWeight = instances.map(_.w).reduce(_ + _)
        val queue = Queue[(Int, List[Instance])]()
        queue += ((root, instances))
        while (!queue.isEmpty) {
            val curObj = queue.dequeue()
            val nodeIndex = curObj._1
            val data = curObj._2
            val node = nodes(nodeIndex)

            // find a best split value on this node
            val leftInstances = data.filter(node.check(_) == true)
            val leftRes = search(leftInstances, totWeight)
            val leftScore = leftRes._1
            val leftSplitVal = leftRes._2
            if (compare(leftScore, minScore) < 0) {
                minScore = leftScore
                bestNode = node
                splitVal = leftSplitVal
                onLeft = true
            }

            val rightInstances = data.filter(node.check(_) == false)
            val rightRes = search(rightInstances, totWeight)
            val rightScore = rightRes._1
            val rightSplitVal = rightRes._2
            if (compare(rightScore, minScore) < 0) {
                minScore = rightScore
                bestNode = node
                splitVal = rightSplitVal
                onLeft = false
            }

            for (c <- node.leftChild) {
                queue += ((c, leftInstances))
            }
            for (c <- node.rightChild) {
                queue += ((c, rightInstances))
            }
        }
        (minScore, (bestNode, onLeft, ThresholdCondition(index, splitVal)))
    }
}
