package sparkboost

import math.log
import math.exp
import util.Random.{nextDouble => rand}
import collection.mutable.ArrayBuffer
import collection.mutable.Queue

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector

import java.io._

import sparkboost.utils.Comparison
import sparkboost.utils.Utils.safeLogRatio

object Type {
    type BaseInstance = (Int, SparseVector)
    type ColRDD = RDD[Instances]
    type BaseRDD = RDD[BaseInstance]
    type BrAI = Broadcast[Array[Int]]
    type BrAD = Broadcast[Array[Double]]
    type BrSV = Broadcast[SparseVector]
    type BrNode = Broadcast[SplitterNode]

    // TODO: here I assumed batchId is always 0 so is ignored
    type BoardKey = (Int, Int, Int, Boolean)
    type BoardType = Map[BoardKey, Double]
    type BrBoard = Broadcast[BoardType]
    type ScoreType = (Double, BoardKey)

    type SampleFunc = Array[SplitterNode] => (RDD[BaseInstance], RDD[BaseInstance])
    type BaseToCSCFunc = RDD[BaseInstance] => ColRDD
    type LossFunc = (Double, Double, Double) => Double
    type Suggest = (Int, Int, Double, Boolean, Double)
    type LearnerObj = (BoardType, ScoreType, ScoreType)
    type LearnerFunc = (SparkContext, ColRDD, BrAI, BrAD,
                        Array[BrSV], Array[BrNode], Int,
                        Double, BrBoard, Int, Int) => LearnerObj
    type UpdateFunc = (ColRDD, BrAI, BrSV, BrAD, BrNode) => (SparseVector, Array[Double])
    type WeightFunc = (Int, Double, Double) => Double
}

class Controller(
    @transient val sc: SparkContext,
    val sampleFunc: Type.SampleFunc,
    val baseToCSCFunc: Type.BaseToCSCFunc,
    val learnerFunc: Type.LearnerFunc,
    val updateFunc: Type.UpdateFunc,
    val lossFunc: Type.LossFunc,
    val weightFunc: Type.WeightFunc,
    val minImproveFact: Double,
    val rawImproveWindow: Int,
    val modelWritePath: String,
    val maxIters: Int
) extends java.io.Serializable with Comparison {
    val printStatsInterval = 10
    val improveWindow = (rawImproveWindow / printStatsInterval).floor.toInt

    var baseTrain: Type.BaseRDD = null
    var train: Type.ColRDD = null
    var y: Type.BrAI = null

    var test: Type.BaseRDD = null
    var testRef: Type.BaseRDD = null

    var depth: Int = 1
    var weights : Type.BrAD = null
    var assign: ArrayBuffer[Broadcast[SparseVector]] = null

    var nodes: Array[Type.BrNode] = null
    var localNodes: Array[SplitterNode] = null
    var lastResample = 0

    // Early stop
    val delta = 0.0001
    val gamma = 0.01
    val kld = (0.5 + gamma) * log((0.5 + gamma) / (0.5 - gamma)) +
                        (0.5 - gamma) * log((0.5 - gamma) / (0.5 + gamma))
    val seqLength = ((1 - delta) * log((1 - delta) / delta) -
                        delta * log(delta / (1 - delta))) / kld
    val seqChunks = (seqLength / 20).ceil.toInt
    val thrA = log(gamma / (1 - gamma))
    val thrB = log((1 - gamma) / gamma)
    val logratio = log((0.5 + gamma) / (0.5 - gamma))

    val trainAvgScores = new Queue[Double]()
    val testAvgScores = new Queue[Double]()

    def setDatasets(baseTrain: Type.BaseRDD, train: Type.ColRDD, y: Type.BrAI,
                    test: Type.BaseRDD, testRef: Type.BaseRDD = null) {
        if (this.baseTrain != null) {
            this.baseTrain.unpersist()
            this.train.unpersist()
            this.y.destroy()
            this.test.unpersist()
        }
        this.baseTrain = baseTrain
        this.train = train
        this.y = y
        this.test = test
        if (testRef != null) {
            if (this.testRef != null) {
                this.testRef.unpersist()
            }
            this.testRef = testRef
        }
    }

    def setNodes(nodes: Array[SplitterNode], lastResample: Int, lastDepth: Int) {
        this.localNodes = nodes
        this.nodes = nodes.map(node => sc.broadcast(node))
        this.depth = lastDepth
    }

    def printStats(iteration: Int) = {
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
            scores.unpersist()
            (sumScores / count, positiveSumScores / positiveCount,
             negativeSumScores / negativeCount, sumScores)
        }

        // Part 1 - Compute auPRC
        val trainPredictionAndLabels = baseTrain.map(t =>
            (SplitterNode.getScore(0, localNodes, t._2).toDouble -
                SplitterNode.getScore(0, localNodes, t._2, lastResample).toDouble, t._1.toDouble)
        ).cache()

        val testPredictionAndLabels = test.map(t =>
            (SplitterNode.getScore(0, localNodes, t._2).toDouble -
                SplitterNode.getScore(0, localNodes, t._2, lastResample).toDouble, t._1.toDouble)
        ).cache()

        val testRefPredictionAndLabels = testRef.map(t =>
            (SplitterNode.getScore(0, localNodes, t._2).toDouble, t._1.toDouble)
        ).cache()

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
            if (iteration % 1 == 0) {
                testRefMetrics = new BinaryClassificationMetrics(testRefPredictionAndLabels)
                auPRCTestRef = testRefMetrics.areaUnderPR + adjust(testRefMetrics.pr.take(2))
                lossFuncTestRef = getLossFunc(testRefPredictionAndLabels)
            } else {
                auPRCTestRef = Double.NaN
                lossFuncTestRef = (Double.NaN, Double.NaN, Double.NaN, Double.NaN)
            }
        }

        println("Training auPRC = " + auPRCTrain)
        println("Training average score = " + lossFuncTrain._1)
        println("Training average score (positive) = " + lossFuncTrain._2)
        println("Training average score (negative) = " + lossFuncTrain._3)
        println("Verify scores and weights " + lossFuncTrain._4 + " " + weights.value.reduce(_ + _))
        println("Testing auPRC = " + auPRCTest)
        println("Testing average score = " + lossFuncTest._1)
        println("Testing average score (positive) = " + lossFuncTest._2)
        println("Testing average score (negative) = " + lossFuncTest._3)
        println("Testing (ref) auPRC = " + auPRCTestRef)
        println("Testing (ref) average score = " + lossFuncTestRef._1)
        println("Testing (ref) average score (positive) = " + lossFuncTestRef._2)
        println("Testing (ref) average score (negative) = " + lossFuncTestRef._3)
        if (iteration % 1000 == 0) {
            println("Training PR = " + trainMetrics.pr.collect.toList)
            println("Testing PR = " + testMetrics.pr.collect.toList)
            println("Testing (ref) PR = " + testRefMetrics.pr.collect.toList)
        }

        // Part 2 - Compute effective counts
        val trainCount = y.value.size
        val positiveTrainCount = y.value.count(_ > 0)
        val negativeTrainCount = trainCount - positiveTrainCount

        val wSum = weights.value.reduce(_ + _)
        val wsqSum = weights.value.map(s => s * s).reduce(_ + _)
        val effectiveCount = (wSum * wSum / wsqSum) / trainCount

        val wPositive = weights.value.zip(y.value).filter(_._2 > 0).map(_._1)
        val wSumPositive = wPositive.reduce(_ + _)
        val wsqSumPositive = wPositive.map(s => s * s).reduce(_ + _)
        val effectiveCountPositive = (wSumPositive * wSumPositive / wsqSumPositive) / positiveTrainCount

        val wSumNegative = wSum - wSumPositive
        val wsqSumNegative = wsqSum - wsqSumPositive
        val effectiveCountNegative = (wSumNegative * wSumNegative / wsqSumNegative) / negativeTrainCount

        println("Effective count = " + effectiveCount)
        println("Positive effective count = " + effectiveCountPositive)
        println("Negative effective count = " + effectiveCountNegative)
        trainPredictionAndLabels.unpersist()
        testPredictionAndLabels.unpersist()
        testRefPredictionAndLabels.unpersist()
        (lossFuncTrain._1, lossFuncTest._1)
    }

    def isUnderfit(avgScore: Double) = {
        trainAvgScores.enqueue(avgScore)
        while (trainAvgScores.size > improveWindow) {
            trainAvgScores.dequeue()
        }
        val improve = (trainAvgScores.head - avgScore) / trainAvgScores.head
        trainAvgScores.size >= improveWindow && (
            compare(minImproveFact) == 0 && compare(improve) <= 0 ||
            compare(improve, minImproveFact) < 0
        )
    }

    def isOverfit(avgScore: Double) = {
        if (testAvgScores.size > 0 && compare(testAvgScores.head, avgScore) >= 0) {
            testAvgScores.clear()
        }
        testAvgScores.enqueue(avgScore)
        testAvgScores.size >= improveWindow
    }

    def overfitRollback() {
        val k = localNodes.size - testAvgScores.size + 1
        println("Rollback due to overfitting from " + localNodes.size + s" nodes to $k nodes")
        (k until nodes.size).foreach { i => nodes(i).destroy() }
        nodes = nodes.take(k)
        localNodes = localNodes.take(k)
        localNodes.foreach{ i => i.child = i.child.filter(_ < k) }
    }

    def setMetaData() {
        assign = new ArrayBuffer[Broadcast[SparseVector]]()
        weights = sc.broadcast((0 until y.value.size).map(_ => 1.0).toArray)
        val fa = sc.broadcast(
            new DenseVector((0 until y.value.size).map(_ => 1.0).toArray).toSparse)
        var nodeIdx = 0
        for (node <- nodes) {
            val faIdx = node.value.prtIndex
            val brFa = if (faIdx < 0) fa else assign(faIdx)
            val (aVec, w) = updateFunc(train, y, brFa, weights, node)
            assign.append(sc.broadcast(aVec))
            if (nodeIdx >= lastResample) {
                val toDestroy = weights
                weights = sc.broadcast(w)
                toDestroy.destroy()
            }
            nodeIdx += 1
        }
        fa.destroy()

        trainAvgScores.clear()
        testAvgScores.clear()
        val (trainAvgScore, testAvgScore) = printStats(0)
        isUnderfit(trainAvgScore)
        isOverfit(testAvgScore)
        println()
    }

    def resample() {
        val (baseTrain, test) = sampleFunc(localNodes)
        val y = sc.broadcast(baseTrain.map(_._1).collect)
        val trainCSC = baseToCSCFunc(baseTrain)
        setDatasets(baseTrain, trainCSC, y, test)
        lastResample = localNodes.size

        println("Train data size: " + baseTrain.count)
        println("Test data size: " + test.count)
        println("Distinct positive samples in the training data: " +
                baseTrain.filter(_._1 > 0).count)
        println("Distinct negative samples in the training data: " +
                baseTrain.filter(_._1 < 0).count)
        println("Distinct positive samples in the test data: " +
                test.filter(_._1 > 0).count)
        println("Distinct negative samples in the test data: " +
                test.filter(_._1 < 0).count)
        println("CSC storage length: " + trainCSC.count)
        println()
    }

    def runADTree(): Array[SplitterNode] = {
        // Initialize the training examples. There are two possible cases:
        //     1. a ADTree is provided (i.e. improve an existing model)
        //     2. start from scratch
        //
        // In both cases, we need to initialize `weights` vector and `assign` matrix.
        // In addition to that, for case 2 we need to create a root node that always says "YES"
        if (nodes == null || nodes.size == 0) {
            val posCount = y.value.count(_ > 0)
            val negCount = y.value.size - posCount
            val predVal = 0.5 * log(posCount.toDouble / negCount)
            val rootNode = SplitterNode(0, -1, 0, (-1, 0.0, true))
            rootNode.setPredict(predVal)

            nodes = Array(sc.broadcast(rootNode))
            localNodes = Array(rootNode)
            println(s"Root node predicts ($predVal, 0.0)")
        }

        setMetaData()

        var curIter = 0
        while (maxIters == 0 || curIter < maxIters) {
            curIter += 1
            println("Node " + localNodes.size)

            val timerStart = System.currentTimeMillis()

            var start = 0
            var board = sc.broadcast(Map[(Int, Int, Int, Boolean), Double]())
            var (minScore, maxScore) = ((0.0, (0, 0, 0, true)), (0.0, (0, 0, 0, true)))
            while (thrA <= minScore._1 && maxScore._1 <= thrB) {
                val res = learnerFunc(
                    sc, train, y, weights, assign.toArray, nodes.toArray, depth,
                    logratio, board, start, seqChunks
                )
                board.destroy()
                board = sc.broadcast(res._1)
                minScore = res._2
                maxScore = res._3
                start = start + seqChunks
            }

            println(s"Stopped after scanning $start examples in (ms) " +
                    (System.currentTimeMillis() - timerStart))

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
            val (nodeIndex, dimIndex, splitIndex, splitEval) = (
                if (compare(thrA - minScore._1, maxScore._1 - thrB) > 0) {
                    minScore._2
                } else {
                    maxScore._2
                }
            )

            // get its prediction
            val curAssign = assign(nodeIndex)
            println("Scores generated on " +
                train.filter(t => t.active && t.index == dimIndex).count + " set(s).")
            val splitVal = train.filter(t => t.active && t.index == dimIndex).first.splits(splitIndex)
            val (posWeight, negWeight) = (
                train.filter(t => t.active && t.index == dimIndex).map(data => {
                    var (posWeight, negWeight) = (0.0, 0.0)
                    (0 until curAssign.value.indices.size).foreach(idx => {
                        val ptr = curAssign.value.indices(idx)
                        if (compare(curAssign.value.values(idx)) != 0 &&
                                (compare(data.x(ptr), splitVal) <= 0) == splitEval) {
                            if (y.value(ptr) > 0) {
                                posWeight += weights.value(ptr)
                                // posCount += 1
                            } else {
                                negWeight += weights.value(ptr)
                                // negCount += 1
                            }
                        }
                    })
                    (posWeight, negWeight)
                }).reduce((a, b) => (a._1 + b._1, a._2 + b._2))
            )
            val pred = 0.5 * safeLogRatio(posWeight, negWeight)

            // add the new node to the nodes list
            val newNode = SplitterNode(nodes.size, nodeIndex, localNodes(nodeIndex).depth + 1,
                                       (dimIndex, splitVal, splitEval))
            newNode.setPredict(pred)
            localNodes(nodeIndex).addChild(localNodes.size)
            val brNewNode = sc.broadcast(newNode)
            nodes :+= brNewNode
            localNodes :+= newNode

            // println(s"weightsAndCounts: ($posWeight, $posCount), ($negWeight, $negCount)")
            println(s"weights: ($posWeight, $negWeight)")
            println("Depth: " + newNode.depth)
            println(s"Predicts $pred. Father $nodeIndex. " +
                    s"Feature $dimIndex, split at $splitVal, eval $splitEval")

            // update weights and assignment matrix
            val timerUpdate = System.currentTimeMillis()
            val (newAssign, newWeights) = updateFunc(train, y, assign(nodeIndex), weights, brNewNode)
            println("updateFunc took (ms) " + (System.currentTimeMillis() - timerUpdate))
            assign.append(sc.broadcast(newAssign))
            println("Changes to weights: " + (newWeights.reduce(_ + _) - weights.value.reduce(_ + _)))
            val toDestroy = weights
            weights = sc.broadcast(newWeights)
            toDestroy.destroy()

            if (curIter % printStatsInterval == 0) {
                SplitterNode.save(localNodes, modelWritePath)
                println("Wrote model to disk at iteration " + curIter)

                val timerPrint = System.currentTimeMillis()
                val (curTrainAvgScore, curTestAvgScore) = printStats(curIter)
                println("printStats took (ms) " + (System.currentTimeMillis() - timerPrint))

                if (isUnderfit(curTrainAvgScore)) {
                    println("Underfitting occurs at iteration " + localNodes.size +
                        s": increasing tree depth from $depth to " + (depth + 1))
                    depth += 1
                    trainAvgScores.clear()
                    testAvgScores.clear()
                    println
                } else if (isOverfit(curTestAvgScore)) {
                    println("Overfitting occurs at iteration " + localNodes.size +
                            ": resampling data")
                    overfitRollback()
                    resample()
                    setMetaData()
                    depth = 1
                    println
                }
            }
            println("Running time for Iteration " + curIter + " is (ms) " +
                    (System.currentTimeMillis() - timerStart))
            println
        }
        localNodes
    }
}
