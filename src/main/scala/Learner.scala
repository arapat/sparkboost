package sparkboost

import collection.mutable.ListBuffer
import collection.mutable.Queue
import util.Random
import Double.MaxValue

import org.apache.spark.rdd.RDD

import utils.Comparison

object Learner extends Comparison {
    type Instance = (Int, Vector[Double], Double)

    def findBestSplit(instances: List[Instance], index: Int, nodes: ListBuffer[SplitterNode], root: Int,
                      lossFunc: (Double, Double, Double, Double, Double) => Double) = {
        def search(curInsts: List[Instance], totWeight: Double) = {
            var minScore = MaxValue
            var splitVal = 0.0

            val posInsts = curInsts.filter(t => compare(t._1, 0.0) > 0)
            val totPos = if (posInsts.size > 0) posInsts.map(_._3).reduce(_ + _) else 0.0
            val negInsts = curInsts.filter(t => compare(t._1, 0.0) < 0)
            val totNeg = if (negInsts.size > 0) negInsts.map(_._3).reduce(_ + _) else 0.0
            val rej = totWeight - totPos - totNeg
            var leftPos = 0.0
            var leftNeg = 0.0
            for (i <- 0 until (curInsts.size - 1)) {
                if (curInsts(i)._1 > 0) {
                    leftPos += curInsts(i)._3
                } else {
                    leftNeg += curInsts(i)._3
                }
                if (compare(curInsts(i)._2(index), curInsts(i + 1)._2(index)) != 0) {
                    val rightPos = totPos - leftPos
                    val rightNeg = totNeg - leftNeg
                    val score = lossFunc(rej, leftPos, leftNeg, rightPos, rightNeg)
                    if (compare(score, minScore) < 0) {
                        minScore = score
                        splitVal = 0.5 * (curInsts(i)._2(index) + curInsts(i + 1)._2(index))
                    }
                }
            }
            (minScore, splitVal)
        }

        var minScore = MaxValue
        var bestNode = nodes(root)
        var splitVal = 0.0
        var onLeft = false

        val totWeight = instances.map(_._3).reduce(_ + _)
        val queue = Queue[(Int, List[Instance])]()
        queue += ((root, instances))
        while (!queue.isEmpty) {
            val curObj = queue.dequeue()
            val nodeIndex = curObj._1
            val data = curObj._2
            val node = nodes(nodeIndex)

            // find a best split value on this node
            val leftInstances = data.filter {t => node.check(t._2) == true}
            val leftRes = search(leftInstances, totWeight)
            val leftScore = leftRes._1
            val leftSplitVal = leftRes._2
            if (compare(leftScore, minScore) < 0) {
                minScore = leftScore
                bestNode = node
                splitVal = leftSplitVal
                onLeft = true
            }

            val rightInstances = data.filter {t => node.check(t._2) == false}
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

    def partitionedGreedySplit(
            instances: RDD[Instance], nodes: ListBuffer[SplitterNode],
            lossFunc: (Double, Double, Double, Double, Double) => Double,
            repartition: Boolean = false, rootIndex: Int = 0) = {
        val inst = instances.first
        val featureSize = inst._2.size
        val shift = Random.nextInt(featureSize)

        def callFindBestSplit(data: (Array[Instance], Long)) = {
            val insts = data._1
            val index = data._2
            val splitIndex: Int = ((index + shift) % featureSize).toInt
            val sortedInsts = insts.toList.sortWith(_._2(splitIndex) < _._2(splitIndex))
            findBestSplit(sortedInsts, splitIndex, nodes, rootIndex, lossFunc)
        }

        val insts = if (repartition) instances.repartition(featureSize) else instances
        val splits = instances.glom().zipWithIndex().map(callFindBestSplit)
        val bestSplit = splits.reduce {(a, b) => if (a._1 < b._1) a else b}
        bestSplit._2
    }
}
