package sparkboost

import math.log
import math.exp
import util.Random.{nextDouble => rand}
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector

import java.io._

import sparkboost.utils.Comparison
import sparkboost.utils.Utils.safeLogRatio

object Controller extends Comparison {
    val R = 30  // TODO: Make this a parameter
    val BINSIZE = 1
    val SEC = 1000000
    type BrAI = Broadcast[Array[Int]]
    type BrAD = Broadcast[Array[Double]]
    type BrSV = Broadcast[SparseVector]
    type BrNode = Broadcast[SplitterNode]
    type RDDType = RDD[Instances]
    type TestRDDType = RDD[(Int, SparseVector)]
    type LossFunc = (Double, Double, Double) => Double
    type SuggestType = (Int, Int, Double, Boolean, Double)
    type LearnerObj = List[SuggestType]
    type LearnerFunc = (SparkContext, RDDType, BrAI, BrAD,
                        Array[BrSV], Array[BrNode], Int, LossFunc) => LearnerObj
    type UpdateFunc = (RDDType, BrAI, BrSV, BrAD, BrNode) => (SparseVector, Array[Double])
    type WeightFunc = (Int, Double, Double) => Double

    def printStats(train: TestRDDType, test: TestRDDType, testRef: TestRDDType,
                   nodes: Array[SplitterNode],
                   y: Array[Int], w: Array[Double],
                   iteration: Int, lastResample: Int) = {
        // manual fix the auPRC computation bug in MLlib
        def adjust(points: Array[(Double, Double)]) = {
            require(points.length == 2)
            require(points.head == (0.0, 1.0))
            val y = points.last
            y._1 * (y._2 - 1.0) / 2.0
        }

        // TODO: extend to other boosting loss functions
        def getLossFunc(predictionAndLabels: RDD[(Double, Double)]) = {
            val scores = predictionAndLabels.map(t => (t._2, exp(-t._1 * t._2))).cache()
            val count = scores.count
            val sumScores = scores.map(_._2).reduce(_ + _)
            val positiveCount = scores.filter(_._1 > 0).count
            val positiveSumScores = scores.filter(_._1 > 0).map(_._2).reduce(_ + _)
            val negativeCount = count - positiveCount
            val negativeSumScores = sumScores - positiveSumScores
            (sumScores / count, positiveSumScores / positiveCount,
             negativeSumScores / negativeCount, sumScores)
        }

        // Part 1 - Compute auPRC
        val trainPredictionAndLabels = train.map {case t =>
            (SplitterNode.getScore(0, nodes, t._2).toDouble -
                SplitterNode.getScore(0, nodes, t._2, lastResample).toDouble, t._1.toDouble)
        }.cache()

        val testPredictionAndLabels = test.map {case t =>
            (SplitterNode.getScore(0, nodes, t._2).toDouble -
                SplitterNode.getScore(0, nodes, t._2, lastResample).toDouble, t._1.toDouble)
        }.cache()

        val testRefPredictionAndLabels = testRef.map {case t =>
            (SplitterNode.getScore(0, nodes, t._2).toDouble, t._1.toDouble)
        }.cache()

        val trainMetrics = new BinaryClassificationMetrics(trainPredictionAndLabels)
        val auPRCTrain = trainMetrics.areaUnderPR + adjust(trainMetrics.pr.take(2))
        val lossFuncTrain = getLossFunc(trainPredictionAndLabels)
        val testMetrics = new BinaryClassificationMetrics(testPredictionAndLabels)
        val auPRCTest = testMetrics.areaUnderPR + adjust(testMetrics.pr.take(2))
        val lossFuncTest = getLossFunc(testPredictionAndLabels)

        var testRefMetrics = testMetrics
        var auPRCTestRef = auPRCTest
        var lossFuncTestRef = lossFuncTest
        if (test.id != testRef.id) {
            testRefMetrics = new BinaryClassificationMetrics(testRefPredictionAndLabels)
            auPRCTestRef = testRefMetrics.areaUnderPR + adjust(testRefMetrics.pr.take(2))
            lossFuncTestRef = getLossFunc(testRefPredictionAndLabels)
        }

        println("Training auPRC = " + auPRCTrain)
        println("Training average score = " + lossFuncTrain._1)
        println("Training average score (positive) = " + lossFuncTrain._2)
        println("Training average score (negative) = " + lossFuncTrain._3)
        println("Verify scores and weights " + lossFuncTrain._4 + " " + w.reduce(_ + _))
        println("Testing auPRC = " + auPRCTest)
        println("Testing average score = " + lossFuncTest._1)
        println("Testing average score (positive) = " + lossFuncTest._2)
        println("Testing average score (negative) = " + lossFuncTest._3)
        println("Testing (ref) auPRC = " + auPRCTestRef)
        println("Testing (ref) average score = " + lossFuncTestRef._1)
        println("Testing (ref) average score (positive) = " + lossFuncTestRef._2)
        println("Testing (ref) average score (negative) = " + lossFuncTestRef._3)
        if (iteration % 20 == 0) {
            println("Training PR = " + trainMetrics.pr.collect.toList)
            println("Testing PR = " + testMetrics.pr.collect.toList)
            println("Testing (ref) PR = " + testRefMetrics.pr.collect.toList)
        }

        // Part 2 - Compute effective counts
        val trainCount = y.size
        val positiveTrainCount = y.count(_ > 0)
        val negativeTrainCount = trainCount - positiveTrainCount

        val wSum = w.reduce(_ + _)
        val wsqSum = w.map(s => s * s).reduce(_ + _)
        val effectiveCount = (wSum * wSum / wsqSum) / trainCount

        val wPositive = w.zip(y).filter(_._2 > 0).map(_._1)
        val wSumPositive = wPositive.reduce(_ + _)
        val wsqSumPositive = wPositive.map(s => s * s).reduce(_ + _)
        val effectiveCountPositive = (wSumPositive * wSumPositive / wsqSumPositive) / positiveTrainCount

        val wSumNegative = wSum - wSumPositive
        val wsqSumNegative = wsqSum - wsqSumPositive
        val effectiveCountNegative = (wSumNegative * wSumNegative / wsqSumNegative) / negativeTrainCount

        println("Effective count = " + effectiveCount)
        println("Positive effective count = " + effectiveCountPositive)
        println("Negative effective count = " + effectiveCountNegative)
        effectiveCount
    }

    def runADTree(sc: SparkContext,
                  train: RDDType, y: Broadcast[Array[Int]],
                  trainRaw: TestRDDType, test: TestRDDType, testRef: TestRDDType,
                  learnerFunc: LearnerFunc,
                  updateFunc: UpdateFunc,
                  lossFunc: LossFunc,
                  weightFunc: WeightFunc,
                  sampleFrac: Double, T: Int, maxDepth: Int,
                  baseNodes: Array[BrNode], writePath: String,
                  lastResample: Int): Array[SplitterNode] = {
        // Report basic meta info about the training data
        val posCount = y.value.count(_ > 0)
        val negCount = y.value.size - posCount
        println(s"Positive examples for training: $posCount")
        println(s"Negative examples for training: $negCount")
        val testPosCount = test.filter(_._1 > 0).count
        val testNegCount = test.count - testPosCount
        println(s"Positive examples for testing: $testPosCount")
        println(s"Negative examples for testing: $testNegCount")

        // Initialize the training examples. There are two possible cases:
        //     1. a ADTree is provided (i.e. improve an existing model)
        //     2. start from scratch
        //
        // In both cases, we need to initialize `weights` vector and `assign` matrix.
        // In addition to that, for case 2 we need to create a root node that always says "YES"
        var nodes =
            if (baseNodes.size == 0) {
                val predVal = 0.5 * log(posCount.toDouble / negCount)
                val rootNode = SplitterNode(0, -1, 0, (-1, 0.0, true))
                rootNode.setPredict(predVal)
                println(s"Root node predicts ($predVal, 0.0)")
                Array(sc.broadcast(rootNode))
            } else {
                baseNodes
            }
        var localNodes = nodes.map(_.value)
        val initAssignAndWeights = {
            val aMatrix = new ArrayBuffer[Broadcast[SparseVector]]()
            var w = sc.broadcast((0 until y.value.size).map(_ => 1.0).toArray)
            val fa = sc.broadcast(
                new DenseVector((0 until y.value.size).map(_ => -1.0).toArray).toSparse)
            var nodeIdx = 0
            for (node <- nodes) {
                val faIdx = node.value.prtIndex
                val brFa = if (faIdx < 0) fa else aMatrix(faIdx)
                val (aVec, nw) = updateFunc(train, y, brFa, w, node)
                aMatrix.append(sc.broadcast(aVec))
                if (nodeIdx >= lastResample) {
                    val toDestroy = w
                    w = sc.broadcast(nw)
                    toDestroy.destroy()
                }
                nodeIdx += 1
            }
            fa.destroy()
            (aMatrix, w)
        }
        val assign = initAssignAndWeights._1
        var weights = initAssignAndWeights._2

        printStats(trainRaw, test, testRef, localNodes, y.value, weights.value, 0, lastResample)
        println()

        def evaluate(t: Instances, suggests: Map[Int, LearnerObj], sumWeight: Double) = {
            suggests(t.index).map(sgst => {
                val (prtNodeIndex, splitIndex, splitVal, splitEval, learnerPredicts) = sgst
                val prtAssign = assign(prtNodeIndex)
                var posWeight = 0.0
                var posCount = 0
                var negWeight = 0.0
                var negCount = 0
                (0 until prtAssign.value.indices.size).foreach(idx => {
                    val ptr = prtAssign.value.indices(idx)
                    if (compare(prtAssign.value.values(idx)) != 0 &&
                        (compare(t.xVec(ptr), splitVal) <= 0) == splitEval) {
                        if (y.value(ptr) > 0) {
                            posWeight += weights.value(ptr)
                            posCount += 1
                        } else {
                            negWeight += weights.value(ptr)
                            negCount += 1
                        }
                    }
                })
                (
                    lossFunc(sumWeight - posWeight - negWeight, posWeight, negWeight),
                    (posWeight, posCount, negWeight, negCount),
                    (prtNodeIndex, splitIndex, splitVal, splitEval, learnerPredicts)
                )
            }).reduce((a, b) => if (a._1 < b._1) a else b)
        }

        var iteration = 0
        while (iteration < math.ceil(T.toDouble / R)) {
            println("Batch " + iteration)
            val timerStart = System.nanoTime()
            iteration = iteration + 1

            // LearnerFunc gives 100 suggestions
            val suggests: Map[Int, LearnerObj] = learnerFunc(
                sc, train, y, weights, assign.toArray, nodes.toArray, maxDepth, lossFunc
            ).groupBy(_._2)
            val pTrain = train.filter(t => suggests.contains(t.index)).cache

            // Iteratively, we select and convert `R` suggestions into weak learners
            (0 until R).foreach(iter2 => {
                val curIter = (iteration - 1) * R + iter2 + 1
                println("Node " + curIter)
                val sumWeight = weights.value.reduce(_ + _)
                val (
                    minScore,
                    (posWeight, posCount, negWeight, negCount),
                    (prtNodeIndex, splitIndex, splitVal, splitEval, learnerPredicts)
                ) = pTrain.map(t => evaluate(t, suggests, sumWeight))
                          .reduce((a, b) => if (a._1 < b._1) a else b)

                /*
                TODO:
                The prediction here is computed based on the statistics collected on just a single
                partition. It is okay when a single partition actually has all training data.
                But if it only has partial data, we may need to add something like below here:
                ```
                    reduce {
                        (a: (Double, Int, Double, Int), b: (Double, Int, Double, Int)) =>
                            (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4)
                    }
                ```
                */

                println(s"weightsAndCounts: ($posWeight, $posCount), ($negWeight, $negCount)")

                val pred = 0.5 * safeLogRatio(posWeight, negWeight)
                println(s"Predicts $pred (suggestion $learnerPredicts) Father $prtNodeIndex")

                // add the new node to the nodes list
                val newNode = SplitterNode(nodes.size, prtNodeIndex, localNodes(prtNodeIndex).depth + 1,
                                           (splitIndex, splitVal, splitEval))
                newNode.setPredict(pred)
                localNodes(prtNodeIndex).addChild(localNodes.size)
                val brNewNode = sc.broadcast(newNode)
                nodes :+= brNewNode
                localNodes :+= newNode

                // update weights and assignment matrix
                val timerUpdate = System.nanoTime()
                val (newAssign, newWeights) = updateFunc(train, y, assign(prtNodeIndex), weights, brNewNode)
                println("updateFunc took (ms) " + (System.nanoTime() - timerUpdate) / SEC)
                assign.append(sc.broadcast(newAssign))
                println("Changes to weights: " + (newWeights.reduce(_ + _) - weights.value.reduce(_ + _)))
                val toDestroy = weights
                weights = sc.broadcast(newWeights)
                toDestroy.destroy()

                val timerStats = System.nanoTime()
                printStats(trainRaw, test, testRef, localNodes, y.value, newWeights, curIter, lastResample)
                println("printStats took (ms) " + (System.nanoTime() - timerUpdate) / SEC)
                println("Running time for Iteration " + iteration + " is (ms) " +
                        (System.nanoTime() - timerStart) / SEC)
                if (iteration % 20 == 0) {
                    SplitterNode.save(localNodes, writePath)
                    println("Wrote model to disk at iteration " + iteration)
                }
                println
            })
            pTrain.unpersist()
        }
        localNodes
    }

    def runADTreeWithAdaBoost(sc: SparkContext,
                              train: RDDType, y: Broadcast[Array[Int]],
                              trainRaw: TestRDDType, test: TestRDDType, testRef: TestRDDType,
                              sampleFrac: Double, T: Int, maxDepth: Int,
                              baseNodes: Array[BrNode], writePath: String,
                              lastResample: Int) = {
        runADTree(sc, train, y, trainRaw, test, testRef,
                  Learner.partitionedGreedySplit, UpdateFunc.adaboostUpdate,
                  LossFunc.lossfunc, UpdateFunc.adaboostUpdateFunc,
                  sampleFrac, T, maxDepth, baseNodes, writePath, lastResample)
    }

    /*
    def runADTreeWithLogitBoost(sc: SparkContext,
                                instances: RDDType, y: Broadcast[Array[Int]], test: TestRDDType,
                                sampleFrac: Double, T: Int, maxDepth: Int,
                                baseNodes: Array[SplitterNode]) = {
        runADTree(sc, instances, y, test,
                  Learner.partitionedGreedySplit, UpdateFunc.logitboostUpdate,
                  LossFunc.lossfunc, UpdateFunc.logitboostUpdateFunc,
                  sampleFrac, T, maxDepth, baseNodes)
    }
    */

    /*
    def runADTreeWithBulkAdaboost(instances: RDD[Instance], T: Int) = {
        runADTree(instances, Learner.bulkGreedySplit, UpdateFunc.adaboostUpdate,
                  LossFunc.lossfunc, T)
    }
    */
}

/*
    def sample(data: RDDType,
               fraction: Double,
               nodes: Array[SplitterNode],
               wfunc: (Int, Double, Double) => Double): RDDType = {
        println("Resampling...")
        val sampleData = data.map(datum => {
            val array = datum._1
            var sampleList = Array[Instance]()
            val size = (array.size * fraction).ceil.toInt
            val weights = array.map(t => wfunc(t.y, 1.0, SplitterNode.getScore(0, nodes, t)))
            val weightSum = weights.reduce(_ + _)
            val segsize = weightSum.toDouble / size

            var curWeight = rand() * segsize // first sample point
            var accumWeight = 0.0
            for (iw <- array.zip(weights)) {
                while (accumWeight <= curWeight && curWeight < accumWeight + iw._2) {
                    sampleList :+= iw._1
                    curWeight += segsize
                }
                accumWeight += iw._2
            }

            (sampleList.map(t => Instance.clone(t, 1.0, nodes)).toList,
             datum._2, datum._3)
        })
        sampleData.checkpoint()
        println("Resampling done. Sample size: " + sampleData.map(_._1.size).reduce(_ + _))
        sampleData
    }
*/
