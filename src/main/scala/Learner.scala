package sparkboost

import collection.mutable.ArrayBuffer
import math.abs
import math.log
import math.min
import math.max
import math.sqrt

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector

import sparkboost.utils.Comparison


object Learner extends Comparison {
    val K = 1

    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("Learner")
    def findBestSplit(
            nodes: Types.ABrNode, maxDepth: Int,
            featuresOffset: Int, featuresPerCore: Int,
            prevScanned: Int, headTest: Int, numTests: Int, thrFact: Double, delta: Double
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
        val rangeSt = (glomId * featuresPerCore + featuresOffset) % numFeatures
        val rangeEd = rangeSt + featuresPerCore
        val range =
            if (rangeEd <= numFeatures) {
                rangeSt until rangeEd
            } else {
                (0 until (rangeEd % numFeatures)) ++ (rangeSt until numFeatures)
            }
        var testStart = headTest % data.size

        // time stamp
        val timeStamp1 = System.currentTimeMillis() - timer

        def findBest(nodeIndex: Int, candid: Array[Int]): (Types.BoardElem, Types.ResultType) = {
            val timer = System.currentTimeMillis()

            val node = nodes(nodeIndex).value
            val (prevw, curScores): (Double, ArrayBuffer[(Double, Double)]) =
                if (board.contains(nodeIndex)) {
                    (board(nodeIndex)._1, ArrayBuffer() ++ board(nodeIndex)._2)
                } else {
                    // TODO: Add support for multiple splits
                    (0.0, ArrayBuffer() ++ range.flatMap(_ => (0 until 1 * 2).map(_ => (0.0, 0.0))))
                }
            var wsum = prevw
            var result = (0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, true)

            val candidIter = candid.iterator
            var earlyStop = false
            while (candidIter.hasNext && !earlyStop) {
                val idx = candidIter.next
                val nScanned =
                    if (idx < testStart) {
                        idx + 1 + data.size - testStart + prevScanned
                    } else {
                        idx - testStart + 1 + prevScanned
                    }
                val (y, x) = data(idx)
                val w = weights(idx)
                wsum += w

                var k = 0
                range.foreach(j => {
                    // TODO: Add support for multiple splits
                    val splitVal = 0.5

                    // Check left tree
                    var (val1, wsum1) = curScores(k)
                    if ((compare(x(j), splitVal) <= 0) == true) {
                        val1 += y * w
                        wsum1 += w
                    }
                    val alpha1 =
                        if (log(wsum1) > delta) {
                            sqrt(K * wsum1 * log(log(wsum1) / delta))
                        } else {
                            Double.MaxValue
                        }
                    val bt1 = abs(val1) - alpha1
                    val gamma1 = bt1 / wsum1

                    // TODO: fix this heuristic
                    if ((wsum1 / wsum) * nScanned >= 100.0 && alpha1 <= bt1 * thrFact) {
                        val result1 = (nScanned, gamma1, val1, wsum1, wsum, nodeIndex, j, 0, true)
                        earlyStop = true
                        result = result1
                    }

                    curScores(k) = (val1, wsum1)
                    k += 1

                    // Check right tree
                    var (val2, wsum2) = curScores(k)
                    if ((compare(x(j), splitVal) <= 0) == false) {
                        val2 += y * w
                        wsum2 += w
                    }
                    val alpha2 =
                        if (log(wsum2) > delta) {
                            sqrt(K * wsum2 * log(log(wsum2) / delta))
                        } else {
                            Double.MaxValue
                        }
                    val bt2 = abs(val2) - alpha2
                    val gamma2 = bt2 / wsum2

                    // TODO: fix this heuristic
                    if ((wsum2 / wsum) * nScanned >= 100.0 && alpha2 <= bt2 * thrFact) {
                        val result2 = (nScanned, gamma2, val2, wsum2, wsum, nodeIndex, j, 0, false)
                        earlyStop = true
                        result = result2
                    }

                    curScores(k) = (val2, wsum2)
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

        val initSamples =
            if (numTests >= data.size) {
                testStart = 0
                (0 until data.size).toArray
            } else if (testStart + numTests > data.size) {
                ((0 until (numTests - (data.size - testStart))).toList ++
                    (testStart until data.size).toList).toArray
            } else {
                (testStart until (testStart + numTests)).toArray
            }
        val (newBoard, bestSplit) = travelTree(0, initSamples)

        (glomId, data, weights,
            newBoard.toMap, bestSplit)
    }

    def partitionedGreedySplit(
            sc: SparkContext, train: Types.TrainRDDType, nodes: Types.ABrNode, maxDepth: Int,
            featuresOffset: Int, featuresPerCore: Int,
            prevScanned: Int, headTest: Int, numTests: Int, thrFact: Double, delta: Double
    ): RDD[Types.GlomResultType] = {
        var tStart = System.currentTimeMillis()

        val f = findBestSplit(nodes, maxDepth,
                              featuresOffset, featuresPerCore,
                              prevScanned, headTest, numTests, thrFact, delta) _
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
