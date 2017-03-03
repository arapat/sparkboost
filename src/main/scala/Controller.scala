package sparkboost

import math.log
import util.Random.{nextDouble => rand}
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.SparseVector

import java.io._

import sparkboost.utils.Comparison
import sparkboost.utils.Utils.safeLogRatio

object Controller extends Comparison {
    val BINSIZE = 1
    type BrAI = Broadcast[Array[Int]]
    type BrAD = Broadcast[Array[Double]]
    type RDDType = RDD[Instances]
    type TestRDDType = RDD[(Int, SparseVector)]
    type LossFunc = (Double, Double, Double, Double, Double) => Double
    type LearnerObj = (Int, Boolean, Int, Double, (Double, Double))
    type LearnerFunc = (RDDType, BrAI, BrAD, Array[BrAI], Array[SplitterNode], Int, LossFunc) => LearnerObj
    type UpdateFunc = (RDDType, BrAI, BrAI, BrAD, SplitterNode) => (Array[Int], Array[Double])
    type WeightFunc = (Int, Double, Double) => Double

    def printStats(train: TestRDDType, test: TestRDDType, nodes: Array[SplitterNode],
                   y: Array[Int], w: Array[Double], iteration: Int) = {
        // manual fix the auPRC computation bug in MLlib
        def adjust(points: Array[(Double, Double)]) = {
            require(points.length == 2)
            require(points.head == (0.0, 1.0))
            val y = points.last
            y._1 * (y._2 - 1.0) / 2.0
        }

        // Part 1 - Compute auPRC
        val trainPredictionAndLabels = train.map {case t =>
            (SplitterNode.getScore(0, nodes, t._2).toDouble, t._1.toDouble)
        }.cache()

        val testPredictionAndLabels = test.map {case t =>
            (SplitterNode.getScore(0, nodes, t._2).toDouble, t._1.toDouble)
        }.cache()

        val trainMetrics = new BinaryClassificationMetrics(trainPredictionAndLabels)
        val auPRCTrain = trainMetrics.areaUnderPR + adjust(trainMetrics.pr.take(2))
        val testMetrics = new BinaryClassificationMetrics(testPredictionAndLabels)
        val auPRCTest = testMetrics.areaUnderPR + adjust(testMetrics.pr.take(2))

        println("Training auPRC = " + auPRCTrain)
        if (iteration % 20 == 0) {
            println("Training PR = " + trainMetrics.pr.collect.toList)
        }
        println("Testing auPRC = " + auPRCTest)

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
                  trainRaw: TestRDDType, test: TestRDDType,
                  learnerFunc: LearnerFunc,
                  updateFunc: UpdateFunc,
                  lossFunc: LossFunc,
                  weightFunc: WeightFunc,
                  sampleFrac: Double, T: Int, maxDepth: Int,
                  baseNodes: Array[SplitterNode]): Array[SplitterNode] = {
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
                val rootNode = SplitterNode(0, -1, true, (-1, 0.0))
                rootNode.setPredict(predVal, 0.0)
                println(s"Root node predicts ($predVal, 0.0)")
                Array(rootNode)
            } else {
                baseNodes
            }
        val initAssignAndWeights = {
            val aMatrix = new ArrayBuffer[Broadcast[Array[Int]]]()
            var w = sc.broadcast((0 until y.value.size).map(_ => 1.0).toArray)
            val fa = sc.broadcast((0 until y.value.size).map(_ => -1).toArray)
            for (node <- nodes) {
                val faIdx = node.prtIndex
                val brFa = if (faIdx < 0) fa else aMatrix(faIdx)
                val (aVec, nw) = updateFunc(train, y, brFa, w, node)
                val toDestroy = w
                w = sc.broadcast(nw)
                aMatrix.append(sc.broadcast(aVec))
                toDestroy.destroy()
            }
            fa.destroy()
            (aMatrix, w)
        }
        val assign = initAssignAndWeights._1
        var weights = initAssignAndWeights._2

        printStats(trainRaw, test, nodes, y.value, weights.value, 0)
        println()

        var iteration = 0
        val SEC = 1000000
        while (iteration < T) {
            val timerStart = System.nanoTime()
            iteration = iteration + 1
            val (prtNodeIndex, onLeft, splitIndex, splitVal, learnerPredicts): LearnerObj =
                    learnerFunc(train, y, weights, assign.toArray, nodes.toArray, maxDepth, lossFunc)
            val newNode = SplitterNode(nodes.size, prtNodeIndex, onLeft, (splitIndex, splitVal))

            val prtAssign = assign(prtNodeIndex)
            // compute the predictions of the new node
            val weightsAndCounts =
                train.filter(_.index == splitIndex).flatMap(t =>
                    t.ptr.zip(t.x.toDense.values) filter {case (k, ix) => {
                        val ia = prtAssign.value(k)
                        ia < 0 && onLeft || ia > 0 && !onLeft
                    }} map {case (k, ix) =>
                        ((ix <= splitVal, y.value(k)), (weights.value(k), 1))
                    }
                ).reduceByKey {
                    (a: (Double, Int), b: (Double, Int)) => (a._1 + b._1, a._2 + b._2)
                }.collectAsMap()
            println("weightsAndCounts:")
            println(weightsAndCounts)

            val leftPositiveWeight = weightsAndCounts.getOrElse((true, 1), (0.0, 0))._1
            val leftNegativeWeight = weightsAndCounts.getOrElse((true, -1), (0.0, 0))._1
            val rightPositiveWeight = weightsAndCounts.getOrElse((false, 1), (0.0, 0))._1
            val rightNegativeWeight = weightsAndCounts.getOrElse((false, -1), (0.0, 0))._1
            val leftPred = 0.5 * safeLogRatio(leftPositiveWeight, leftNegativeWeight)
            val rightPred = 0.5 * safeLogRatio(rightPositiveWeight, rightNegativeWeight)
            println(s"Predicts ($leftPred, $rightPred) Father $prtNodeIndex")

            // add the new node to the nodes list
            newNode.setPredict(leftPred, rightPred)
            nodes(prtNodeIndex).addChild(onLeft, nodes.size)
            nodes :+= newNode

            // update weights and assignment matrix
            val timerUpdate = System.nanoTime()
            val (newAssign, newWeights) = updateFunc(train, y, prtAssign, weights, newNode)
            println("updateFunc took (ms) " + (System.nanoTime() - timerUpdate) / SEC)
            assign.append(sc.broadcast(newAssign))
            val toDestroy = weights
            weights = sc.broadcast(newWeights)
            toDestroy.destroy()

            val timerStats = System.nanoTime()
            printStats(trainRaw, test, nodes, y.value, newWeights, iteration)
            println("printStats took (ms) " + (System.nanoTime() - timerUpdate) / SEC)
            println("Running time for Iteration " + iteration + " is (ms) " +
                    (System.nanoTime() - timerStart) / SEC)
            println
        }
        nodes
    }

    def runADTreeWithAdaBoost(sc: SparkContext,
                              train: RDDType, y: Broadcast[Array[Int]],
                              trainRaw: TestRDDType, test: TestRDDType,
                              sampleFrac: Double, T: Int, maxDepth: Int,
                              baseNodes: Array[SplitterNode]) = {
        runADTree(sc, train, y, trainRaw, test,
                  Learner.partitionedGreedySplit, UpdateFunc.adaboostUpdate,
                  LossFunc.lossfunc, UpdateFunc.adaboostUpdateFunc,
                  sampleFrac, T, maxDepth, baseNodes)
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
