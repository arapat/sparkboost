package sparkboost

import math.abs
import math.log
import math.exp
import math.min
import math.pow
import util.Random.{nextDouble => rand}
import util.Random.{nextInt => randomInt}
import collection.mutable.ArrayBuffer
import collection.mutable.Queue

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector

import java.io._

import sparkboost.utils.Comparison
import sparkboost.utils.Utils

class Controller(
    @transient val sc: SparkContext,
    val sampleFunc: Types.SampleFunc,
    val learnerFunc: Types.LearnerFunc,
    val updateFunc: Types.UpdateFunc,
    val minImproveFact: Double,
    val rawImproveWindow: Int,
    val modelWritePath: String,
    val maxIters: Int
) extends java.io.Serializable with Comparison {
    val printStatsInterval = 20
    val emptyMap = Map[Int, (Double, Array[Double])]()

    var train: Types.BaseRDD = null
    var glomTrain: Types.TrainRDDType = null
    var test: Types.BaseRDD = null
    var testRef: Types.BaseRDD = null
    var maxPartSize: Int = 0

    var depth: Int = 100

    var nodes: ArrayBuffer[Types.BrNode] = null
    var localNodes: Array[SplitterNode] = null
    var lastResample = 0

    // Early stop
    var gamma = 0.25
    var delta = pow(10, -20)
    val initSeqChunks = 8000
    var seqChunks = initSeqChunks

    // TT: good
    def setDatasets(train: Types.BaseRDD, test: Types.BaseRDD, testRef: Types.BaseRDD = null) {
        if (this.train != null) {
            this.train.unpersist()
            this.test.unpersist()
        }
        this.train = train
        // TODO: weights are not corrected for pre-loaded models
        glomTrain = train.glom()
                         .zipWithIndex()
                         .map { case (array, idx) => {
                             (idx.toInt, array, array.map(_ => 1.0), emptyMap)
                         }}.cache()
        this.test = test
        if (testRef != null) {
            if (this.testRef != null) {
                this.testRef.unpersist()
            }
            this.testRef = testRef
        }

        glomTrain.count
        maxPartSize = glomTrain.map(_._2.size).max
        println(s"Max partition size is $maxPartSize")
        println
    }

    def setGlomTrain(glomResults: Types.ResultRDDType = null) {
        val toDestroy = glomTrain
        if (glomResults == null) {
            glomTrain = glomTrain.map(t => (t._1, t._2, t._3, emptyMap)).cache
        } else {
            glomTrain = glomResults.map(t => (t._1, t._2, t._3, t._4)).cache
        }
        glomTrain.count
        toDestroy.unpersist()
    }

    // TT: good
    def setNodes(nodes: Array[SplitterNode], lastResample: Int, lastDepth: Int) {
        this.localNodes = nodes
        this.nodes = ArrayBuffer() ++ nodes.map(node => sc.broadcast(node))
        // this.depth = lastDepth
    }

    // TT: good
    def resample() {
        val (train, test) = sampleFunc(localNodes)
        train.setName("sampled train data")
        test.setName("sampled test data")

        setDatasets(train, test)
        lastResample = localNodes.size

        println("Train data size: " + train.count)
        println("Test data size: " + test.count)
        println("Distinct positive samples in the training data: " +
                train.filter(_._1 > 0).count)
        println("Distinct negative samples in the training data: " +
                train.filter(_._1 < 0).count)
        println("Distinct positive samples in the test data: " +
                test.filter(_._1 > 0).count)
        println("Distinct negative samples in the test data: " +
                test.filter(_._1 < 0).count)
        println()
    }

    def runADTree(): Array[SplitterNode] = {
        def maxAbs(a: Double, b: Double) = if (abs(a) > abs(b)) a else b
        def safeMaxAbs(n: Int)(value: (Double, Array[Double])) = {
            val (wsum, array) = value
            if (array.size == 0) 0.0 else array.reduce(maxAbs)
        }
        def safeMaxAbs2(array: Iterator[Double]) = if (array.hasNext == false) 0.0 else array.reduce(maxAbs)
        def safeMaxAbs3(array: RDD[Double]) = if (array.take(1).size == 0) 0.0 else array.reduce(maxAbs)
        // Initialize the training examples. There are two possible cases:
        //     1. a ADTree is provided (i.e. improve an existing model)
        //     2. start from scratch
        //
        // In both cases, we need to initialize `weights` vector and `assign` matrix.
        // In addition to that, for case 2 we need to create a root node that always says "YES"
        if (nodes == null || nodes.size == 0) {
            val posCount = train.filter(_._1 > 0).count
            val negCount = train.count - posCount
            val predVal = 0.5 * log(posCount.toDouble / negCount)
            val rootNode = SplitterNode(0, -1, 0, (-1, 0.0, true))
            rootNode.setPredict(predVal)

            nodes = ArrayBuffer(sc.broadcast(rootNode))
            localNodes = Array(rootNode)
            println(s"Root node predicts ($predVal, 0.0)")
        }

        // TODO: is this func still needed?
        // setMetaData()

        var curIter = 0
        var terminate = false
        while (!terminate && (maxIters == 0 || curIter < maxIters)) {
            val timerStart = System.currentTimeMillis()

            curIter += 1
            println("Node " + localNodes.size)

            // 1. Find a good weak rule
            //    Simulating TMSN using Spark's computation model ==>
            //        Ask workers to scan a batch of examples at a time until one of them find
            //        a valid weak rule (as per early stop rule).
            var resSplit: Types.ResultType = (0, 0.0, 0, 0, 0, true)

            while (gamma > 0.02 && resSplit._1 == 0) {
                var start = 0
                var scanned = 0
                setGlomTrain()
                while (scanned < maxPartSize && resSplit._1 == 0) {
                    val thrFunc = Utils.getThreshold(gamma, delta) _
                    println(s"Now scan $seqChunks examples from $start for a $gamma weak learner.")
                    // TODO: let 0, 8 be two parameters
                    val glomResults = learnerFunc(
                        sc, glomTrain, nodes, depth,
                        0, 8,
                        scanned, start, seqChunks, thrFunc
                    ).cache()
                    val results = glomResults.map(_._5).filter(_._1 != 0).cache()
                    if (results.count > 0) {
                        // for simulation: select the earliest stopped one
                        resSplit = results.reduce((a, b) => if (a._1 < b._1) a else b)
                    } else {
                        setGlomTrain(glomResults)
                    }
                    start += seqChunks
                    scanned += seqChunks

                    println("We have " + results.count + " potential biased rules.")

                    {
                        // Debug
                        setGlomTrain(glomResults)
                        // println(glomResults.first._4.keys.toList.take(5).toList)
                        // println(glomResults.first._4.values.toList.head.toList)
                        // println(glomTrain.first._4)
                    }

                    glomResults.unpersist()
                    println("Testing progress: most extreme outlier " +
                        safeMaxAbs3(glomTrain.map(t =>
                            safeMaxAbs2(t._4.values.map(safeMaxAbs(scanned)).toIterator)
                        )) + ", threshold " + thrFunc(scanned))
                }
                seqChunks = start
                if (resSplit._1 == 0) {
                    println(s"=== !!!  Cannot find a valid weak learner for $gamma.  !!! ===")
                    gamma /= 2.0
                    seqChunks = initSeqChunks
                }
            }

            if (resSplit._1 == 0) {
                println("=== !!!  Cannot find a valid weak learner at all.  !!! ===")
                return localNodes
            }

            println(s"Stopped after scanning $seqChunks examples in (ms) " +
                    (System.currentTimeMillis() - timerStart))

            val (steps, score, nodeIndex, dimIndex, splitIndex, splitEval) = resSplit
            val splitVal = 0.5  // TODO: fix this
            val pred = if (score > 0) (0.5 * log((0.5 + gamma) / (0.5 - gamma)))
                       else           (0.5 * log((0.5 - gamma) / (0.5 + gamma)))

            println(s"$steps steps achieved score $score, threshold was " +
                Utils.getThreshold(gamma, delta)(steps))

            // add the new node to the nodes list
            val newNodeId = nodes.size
            val newNode = SplitterNode(newNodeId, nodeIndex, localNodes(nodeIndex).depth + 1,
                                       (dimIndex, splitVal, splitEval))
            newNode.setPredict(pred)
            localNodes(nodeIndex).addChild(localNodes.size)
            val brNewNode = sc.broadcast(newNode)
            nodes :+= brNewNode
            localNodes :+= newNode
            println("Depth: " + newNode.depth)
            println(s"Predicts $pred. Father $nodeIndex. " +
                    s"Feature $dimIndex, split at $splitVal, eval $splitEval")

            {
                // Debug
                def check(x: SparseVector, nodeId: Int): Boolean = {
                    val node = nodes(nodeId).value
                    ((node.prtIndex < 0 || check(x, node.prtIndex)) &&
                        (node.splitIndex < 0 || node.check(x(node.splitIndex))))
                }

                val (pos, neg) = glomTrain.map(t => {
                    var i = 0
                    var (pos, neg) = (0.0, 0.0)
                    if (t._3.size >= steps) {
                        while (i < steps) {
                            val (y, x) = t._2(i)
                            val w = t._3(i)
                            if (check(x, newNodeId)) {
                                if (y > 0) {
                                    pos += w
                                } else {
                                    neg += w
                                }
                            }
                            i+= 1
                        }
                        (true, (pos, neg))
                    } else {
                        (false, (pos, neg))
                    }
                }).filter(_._1).map(_._2).reduce((a, b) =>
                    (a._1 + b._1, a._2 + b._2)
                )
                println("Actual prediction should be " + 0.5 * log(pos / neg))
            }

            glomTrain = updateFunc(glomTrain, nodes)
            glomTrain.count

            if (curIter % printStatsInterval == 0) {
                SplitterNode.save(localNodes, modelWritePath)
                println("Wrote model to disk at iteration " + curIter)

                val timerPrint = System.currentTimeMillis()
                val (curTrainAvgScore, curTestAvgScore) = Utils.printStats(
                    train, glomTrain, test, testRef, localNodes, curIter
                )
                println("printStats took (ms) " + (System.currentTimeMillis() - timerPrint))
            }

            println("Running time for Iteration " + curIter + " is (ms) " +
                    (System.currentTimeMillis() - timerStart))
            println
        }
        localNodes
    }
}

/*
    val trainAvgScores = new Queue[Double]()
    val testAvgScores = new Queue[Double]()

    def setMetaData() {
        trainAvgScores.clear()
        testAvgScores.clear()
        val (trainAvgScore, testAvgScore) = printStats(0)
        isUnderfit(trainAvgScore)
        isOverfit(testAvgScore)
        println()
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

        if (isUnderfit(curTrainAvgScore)) {
            println("Underfitting occurs at iteration " + localNodes.size +
                s": increasing tree depth from $depth to " + (depth + 1))
            depth += 1
            // trainAvgScores.clear()
            // testAvgScores.clear()
            println
        } else if (isOverfit(curTestAvgScore)) {
            println("Overfitting occurs at iteration " + localNodes.size +
                    ": resampling data")
            overfitRollback()
            resample()
            setMetaData()
            depth = 100
            println
        }
*/
