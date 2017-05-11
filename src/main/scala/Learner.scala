package sparkboost

import collection.mutable.ArrayBuffer
import math.abs
import math.min
import math.max
import math.sqrt

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector

import sparkboost.utils.Comparison


object ZeroIterator extends Iterator[Double] {
    def hasNext = true
    def next = 0.0
}


object Learner extends Comparison {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("Learner")

    def findBestSplit(
            nodes: Types.ABrNode, maxDepth: Int,
            featuresOffset: Int, featuresPerCore: Int,
            headTest: Int, numTests: Int, getThreshold: Int => Double
    )(glom: Types.GlomType): Types.GlomResultType = {

        // TODO:
        // Is this better than serialize the whole tree and pass it as part of the function?
        def getTreeTopology() = {
            val res = ArrayBuffer() ++ (0 until nodes.size).map(_ => Array[Int]())
            (0 until nodes.size).foreach(i => {
                val prt = nodes(i).value.prtIndex
                if (prt >= 0)
                    res(prt) = i +: res(prt)
            })
            res.toArray
        }

        val timer = System.currentTimeMillis()

        val (glomId, data, weights, board) = glom
        val numFeatures = data(0)._2.size
        val tree = getTreeTopology()
        val rangeSt = glomId + featuresOffset
        val rangeEd = rangeSt + featuresPerCore
        val range =
            if (rangeEd <= numFeatures) {
                rangeSt until rangeEd
            } else {
                (0 until (rangeEd % numFeatures)) ++ (rangeSt until numFeatures)
            }

        // time stamp
        val timeStamp1 = System.currentTimeMillis() - timer

        def findBest(nodeIndex: Int, candid: Array[Int]): (Types.BoardElem, Types.ResultType) = {
            def getSign(t: Double, value: Int) = {
                if (compare(t) > 0) value else -value
            }

            val timer = System.currentTimeMillis()

            val node = nodes(nodeIndex).value
            val prevScores = if (board.contains(nodeIndex)) board(nodeIndex).iterator else ZeroIterator
            var curScores = Array[Double]()
            var result = (0, 0, 0, 0, true)

            var wsum = 0.0
            val candidIter = candid.iterator
            var earlyStop = false
            while (candidIter.hasNext && !earlyStop) {
                val idx = candidIter.next
                val nScanned = idx - headTest + 1
                val thr = getThreshold(nScanned)
                val (y, x) = data(idx)
                val w = weights(idx)
                wsum += w
                val score = y * w

                range.foreach(j => {
                    // TODO: Added support to multiple splits
                    val splitVal = 0.5

                    // Check left tree
                    val val1 = prevScores.next + (
                        if ((compare(x(j), splitVal) <= 0) == true) {
                            score
                        } else {
                            0.0
                        }
                    )
                    val result1 = (getSign(val1, nScanned), nodeIndex, j, 0, true)
                    if (abs(val1) > thr) {
                        earlyStop = true
                        result = result1
                    }
                    curScores = val1 +: curScores

                    // Check right tree
                    val val2 = prevScores.next + (
                        if ((compare(x(j), splitVal) <= 0) == false) {
                            score
                        } else {
                            0.0
                        }
                    )
                    val result2 = (getSign(val2, nScanned), nodeIndex, j, 0, false)
                    if (abs(val2) > thr) {
                        earlyStop = true
                        result = result2
                    }
                    curScores = val2 +: curScores
                })
            }

            ((nodeIndex, curScores.reverse), result)
        }

        def travelTree(nodeId: Int, faCandid: Array[Int]): (Types.BoardList, Types.ResultType) = {
            val node = nodes(nodeId).value
            val child = tree(nodeId)
            val candid = faCandid.filter(i =>
                node.check(data(i)._2(max(0, node.splitIndex)))
            )

            var (cb, bestSplit) = findBest(nodeId, candid)
            var newBoard = List(cb)

            if (node.depth + 1 < maxDepth) {
                val c = child.iterator
                while (bestSplit._1 == 0 && c.hasNext) {
                    val (res, split) = travelTree(c.next, candid)
                    newBoard ++= res
                    bestSplit = split
                }
            }

            (newBoard, bestSplit)
        }

        val totTests = min(headTest + numTests, data.size)
        val (newBoard, bestSplit) = travelTree(0, (headTest until totTests).toArray)

        (glomId, data, weights,
            newBoard.toMap, bestSplit)
    }

    def partitionedGreedySplit(
            sc: SparkContext, train: Types.TrainRDDType, nodes: Types.ABrNode, maxDepth: Int,
            featuresOffset: Int, featuresPerCore: Int,
            headTest: Int, numTests: Int, getThreshold: Int => Double
    ): RDD[Types.GlomResultType] = {
        var tStart = System.currentTimeMillis()

        val f = findBestSplit(nodes, maxDepth,
                              featuresOffset, featuresPerCore,
                              headTest, numTests, getThreshold) _
        val trainAndResult = train.map(f).cache()
        trainAndResult.count()

        println("FindWeakLearner took in total (ms) " + (System.currentTimeMillis() - tStart))
        trainAndResult
    }
}


/*
    def getLossScore(score: Double, key: BoardKey) = {
        if (thrA <= score && score <= thrB) {
            (Double.NaN, Double.NaN)
        } else {
            val (nodeIndex, dim, splitIdx, splitEval) = key
            // TODO: fix this
            val splitVal = 0.5

            var (posWeight, negWeight) = (0.0, 0.0)
            (0 until curAssign.indices.size).map(idx => {
                val ptr = curAssign.indices(idx)
                val loc = curAssign.values(idx)
                if (compare(loc) != 0) {
                    val iy = y.value(ptr)
                    if ((compare(data._3(ptr), splitVal) <= 0) == splitEval) {
                        if (y.value(ptr) > 0) {
                            posWeight += w.value(ptr)
                        } else {
                            negWeight += w.value(ptr)
                        }
                    }
                }
            })
            (
                (wsum - posWeight - negWeight) + 2 * sqrt(posWeight * negWeight),
                posWeight / negWeight
            )
        }
    }
*/
