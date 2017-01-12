package sparkboost

import collection.mutable.ListBuffer
import collection.mutable.Queue
import util.Random
import Double.MaxValue

import org.apache.spark.rdd.RDD

import utils.Comparison

object Learner extends Comparison {
    @transient lazy val log = org.apache.log4j.LogManager.getLogger("Learner")

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
            log.info("Processing " + curInsts.size + " instances on the index " + index + ".")
            var lastVal = if (curInsts.size > 0) curInsts(0).X(index) else 0.0
            for (t <- curInsts) {
                if (t.y > 0) {
                    leftPos += t.w
                } else {
                    leftNeg += t.w
                }
                if (compare(t.X(index), lastVal) != 0) {
                    val rightPos = totPos - leftPos
                    val rightNeg = totNeg - leftNeg
                    val score = lossFunc(rej, leftPos, leftNeg, rightPos, rightNeg)
                    if (compare(score, minScore) < 0) {
                        minScore = score
                        splitVal = 0.5 * (t.X(index) + lastVal)
                    }
                }
                lastVal = t.X(index)
            }
            log.info("Index " + index + " is processed.")
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
            log.info("Running on the left of node " + nodeIndex)
            val leftInstances = data.filter {_.scores(nodeIndex) > 0}
            log.info("Left instances size: " + leftInstances.size)
            val leftRes = search(leftInstances, totWeight)
            val leftScore = leftRes._1
            val leftSplitVal = leftRes._2
            if (compare(leftScore, minScore) < 0) {
                minScore = leftScore
                bestNodeIndex = nodeIndex
                splitVal = leftSplitVal
                onLeft = true
            }

            log.info("Running on the right of node " + nodeIndex)
            val rightInstances = data.filter {_.scores(nodeIndex) < 0}
            log.info("Right instances size: " + rightInstances.size)
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
            instances: RDD[Instance], nodes: ListBuffer[SplitterNode],
            lossFunc: (Double, Double, Double, Double, Double) => Double,
            repartition: Boolean = true, rootIndex: Int = 0) = {
        val inst = instances.first
        val featureSize = inst.X.size
        val shift = Random.nextInt(featureSize)

        def callFindBestSplit(data: (Array[Instance], Long)) = {
            val insts = data._1
            val index = data._2
            val splitIndex: Int = ((index + shift) % featureSize).toInt
            val sortedInsts = insts.toList.sortWith(_.X(splitIndex) < _.X(splitIndex))
            findBestSplit(sortedInsts, splitIndex, nodes, rootIndex, lossFunc)
        }

        // val insts = if (repartition) instances.repartition(featureSize) else instances
        val splits = instances.glom().zipWithIndex().map(callFindBestSplit)
        val bestSplit = splits.reduce {(a, b) => if (a._1 < b._1) a else b}
        println("Min score is " + bestSplit._1)
        bestSplit._2
    }

    def bulkGreedySplit(
            instances: RDD[Instance], nodes: ListBuffer[SplitterNode],
            lossFunc: (Double, Double, Double, Double, Double) => Double,
            repartition: Boolean = false, rootIndex: Int = 0) = {
        val inst = instances.first
        val featureSize = inst.X.size
        val insts = instances.collect()
        val splits = (0 until featureSize) map (i => {
            findBestSplit(insts.sortWith(_.X(i) < _.X(i)).toList, i, nodes, rootIndex, lossFunc)
        })
        val bestSplit = splits.reduce {(a, b) => if (a._1 < b._1) a else b}
        println("Min score is " + bestSplit._1)
        bestSplit._2
    }
}
