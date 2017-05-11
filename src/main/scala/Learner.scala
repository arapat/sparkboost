package sparkboost

import collection.mutable.ArrayBuffer
import math.min
import math.max
import math.sqrt

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector

import sparkboost.Types
import sparkboost.utils.Comparison

object Learner extends Comparison {
    // @transient lazy val log = org.apache.log4j.LogManager.getLogger("Learner")

    def findBestSplit(
            nodes: ABrNode, maxDepth: Int,
            featuresOffset: Int, featuresPerCore: Int,
            headTest: Int, numTests: Int, thrA: Double, thrB: Double
    )(glom: GlomType): GlomResultType = {

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
        val rangeSt = partitionIndex + featuresOffset
        val rangeEd = rangeSt + featuresPerCore
        val range =
            if (rangeEd <= numFeatures) {
                rangeSt until rangeEd
            } else {
                (0 until (rangeEd % numFeatures)) ++ (rangeSt until numFeatures)
            }

        // time stamp
        val timeStamp1 = System.currentTimeMillis() - timer

        def findBest(nodeIndex: Int, candid: Array[Int]): (BoardElem, ResultType) = {
            def nextOrElse(iter: Iterator[Double]) = {
                if (iter.hasNext) {
                    iter.next
                } else {
                    0.0
                }
            }

            val timer = System.currentTimeMillis()

            val node = nodes(nodeIndex).value
            val prevScores = board(nodeIndex).iterator
            var curScores = Array[Double]()
            var result = (-1, 0, 0, 0, true)

            var wsum = 0.0
            val candidIter = candid.iterator
            var earlyStop = false
            while (candidIter.hasNext && !earlyStop) {
                val idx = candidIter.next
                val (y, x) = data(idx)
                val w = weights(idx)
                wsum += w
                val score = y * w

                range.foreach(j => {
                    // TODO: Added support to multiple splits
                    val splitVal = 0.5

                    // Check left tree
                    val result1 = (idx - headTest + 1, nodeIndex, j, 0, true)
                    val val1 = nextOrElse(prevScores) +
                        if ((compare(x(j), splitVal) <= 0) == true) {
                            score
                        } else {
                            0.0
                        }
                    )
                    if (val1 < thrA || val1 > thrB) {
                        earlyStop = true
                        result = result1
                    }
                    curScores = val1 +: curScores

                    // Check right tree
                    val result2 = (idx - headTest + 1, nodeIndex, j, 0, false)
                    val val2 = nextOrElse(prevScores) +
                        if ((compare(x(j), splitVal) <= 0) == false) {
                            score
                        } else {
                            0.0
                        }
                    )
                    if (val2 < thrA || val2 > thrB) {
                        earlyStop = true
                        result = result2
                    }
                    curScores = val2 +: curScores
                })
            }

            ((nodeIndex, curScores), result)
        }

        def travelTree(nodeId: Int, faCandid: Array[Int]): (BoardList, ResultType) = {
            val node = nodes(nodeId).value
            val child = travelTree(nodeId)
            val candid = faCandid.filter(i =>
                node.check(data(i)._2(max(0, node.splitIndex)),
                           node.splitIndex, true)
            )

            var (cb, bestSplit) = findBest(j, candid)
            var newBoard = List(cb)

            if (node.depth + 1 < maxDepth) {
                val c = child.iterator
                while (bestSplit._1 < 0 && c.hasNext)
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
            sc: SparkContext, train: TrainRDDType, nodes: ABrNode, maxDepth: Int,
            featuresOffset: Int, featuresPerCore: Int,
            headTest: Int, numTests: Int, thrA: Double, thrB: Double
    ): RDD[GlomResultType] = {
        var tStart = System.currentTimeMillis()

        val f = findBestSplit(nodes, maxDepth,
                              featuresOffset, featuresPerCore,
                              headTest, numTests, thrA, thrB) _
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
