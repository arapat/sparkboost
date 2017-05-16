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


object Learner extends Comparison {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("Learner")

    def findBestSplit(
            nodes: Types.ABrNode, maxDepth: Int,
            featuresOffset: Int, featuresPerCore: Int,
            prevScanned: Int, headTest: Int, numTests: Int, getThreshold: Double => Double
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
        var testStart = headTest % data.size
        testStart = if (testStart + numTests > data.size) max(0, data.size - numTests) else testStart
        val totTests = min(testStart + numTests, data.size)

        // time stamp
        val timeStamp1 = System.currentTimeMillis() - timer

        def findBest(nodeIndex: Int, candid: Array[Int]): (Types.BoardElem, Types.ResultType) = {
            val timer = System.currentTimeMillis()

            val node = nodes(nodeIndex).value
            val (prevw, curScores): (Double, ArrayBuffer[Double]) =
                if (board.contains(nodeIndex)) {
                    (board(nodeIndex)._1, ArrayBuffer() ++ board(nodeIndex)._2)
                } else {
                    // TODO: Add support for multiple splits
                    (0.0, ArrayBuffer() ++ range.flatMap(_ => (0 until 1 * 2).map(_ => 0.0)))
                }
            var wsum = prevw
            var result = (0, 0.0, 0.0, 0, 0, 0, true)

            val candidIter = candid.iterator
            var earlyStop = false
            while (candidIter.hasNext && !earlyStop) {
                val idx = candidIter.next
                val nScanned = idx - testStart + 1 + prevScanned
                val (y, x) = data(idx)
                val w = weights(idx)
                wsum += w
                val thr = getThreshold(wsum)
                val score = y * w

                var k = 0
                range.foreach(j => {
                    // TODO: Add support for multiple splits
                    val splitVal = 0.5

                    // Check left tree
                    val val1 = curScores(k) + (
                        if ((compare(x(j), splitVal) <= 0) == true) {
                            score
                        } else {
                            0.0
                        }
                    )
                    val result1 = (nScanned, val1, wsum, nodeIndex, j, 0, true)
                    if (abs(val1) > thr) {
                        earlyStop = true
                        result = result1
                    }
                    curScores(k) = val1
                    k += 1

                    // Check right tree
                    val val2 = curScores(k) + (
                        if ((compare(x(j), splitVal) <= 0) == false) {
                            score
                        } else {
                            0.0
                        }
                    )
                    val result2 = (nScanned, val2, wsum, nodeIndex, j, 0, false)
                    if (abs(val2) > thr) {
                        earlyStop = true
                        result = result2
                    }
                    curScores(k) = val2
                    k += 1
                })
            }

            ((nodeIndex, (wsum, curScores.toArray)), result)
        }

        def travelTree(nodeId: Int, faCandid: Array[Int]): (Types.BoardList, Types.ResultType) = {
            val node = nodes(nodeId).value
            val child = tree(nodeId)
            val candid = faCandid.filter(i =>
                node.check(data(i)._2(max(0, node.splitIndex)))
            )

            var (cb, bestSplit) = findBest(nodeId, candid)
            var newBoard: List[Types.BoardElem] = List(cb)

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

        val (newBoard, bestSplit) = travelTree(0, (headTest until totTests).toArray)

        (glomId, data, weights,
            newBoard.toMap, bestSplit)
    }

    def partitionedGreedySplit(
            sc: SparkContext, train: Types.TrainRDDType, nodes: Types.ABrNode, maxDepth: Int,
            featuresOffset: Int, featuresPerCore: Int,
            prevScanned: Int, headTest: Int, numTests: Int, getThreshold: Double => Double
    ): RDD[Types.GlomResultType] = {
        var tStart = System.currentTimeMillis()

        val f = findBestSplit(nodes, maxDepth,
                              featuresOffset, featuresPerCore,
                              prevScanned, headTest, numTests, getThreshold) _
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
