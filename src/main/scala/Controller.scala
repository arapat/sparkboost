package sparkboost

import math.abs
import math.log
import math.exp
import math.min
import math.max
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
    val clearCheckpoints: () => Unit,
    val minImproveFact: Double,
    val rawImproveWindow: Int,
    val modelWritePath: String,
    val maxIters: Int,
    val numCores: Int
) extends java.io.Serializable with Comparison {
    val EPS = 0.001

    val printStatsInterval = 100
    val emptyMap: Types.BoardType = Map[Int, (Double, Array[Types.BoardInfo])]()

    var train: Types.BaseRDD = null
    var glomTrain: Types.TrainRDDType = null
    var test: Types.BaseRDD = null
    var testRef: Types.BaseRDD = null
    var maxPartSize: Int = 0
    var minPartSize: Int = 0

    var nodes: ArrayBuffer[Types.BrNode] = null
    var localNodes: Array[SplitterNode] = null
    var lastResample = 0

    var checkpoint = 0

    // Early stop
    val INIT_GAMMA = 0.25
    val MIN_GAMMA = 0.0
    var delta = pow(10, -2) / 600
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
        // TODO: hard coded the number of partitions
        println(s"We will be using $numCores cores.")
        train.checkpoint()
        train.count()
        glomTrain = train.repartition(numCores).glom()
                         .zipWithIndex()
                         .map { case (array, idx) => {
                             (idx.toInt, array, array.map(_ => 1.0), 1.0, emptyMap)
                         }}.cache()
        glomTrain.setName("glomTrain in setDatasets")

        {
            // glomTrain.map(_._2.slice(0, 20).map(_._1).toList).collect().foreach{t => println(t.toList)}
        }

        this.test = test
        if (testRef != null) {
            if (this.testRef != null) {
                this.testRef.unpersist()
            }
            this.testRef = testRef
        }

        maxPartSize = glomTrain.map(_._2.size).max
        minPartSize = glomTrain.map(_._2.size).min
        println("Number of partitions: " + glomTrain.count)
        println(s"Max partition size is $maxPartSize")
        println(s"Min partition size is $minPartSize")
        println
    }

    def setGlomTrain(glomResults: Types.ResultRDDType = null) {
        val toDestroy = glomTrain
        if (glomResults == null) {
            glomTrain = glomTrain.map(t => (t._1, t._2, t._3, t._4, emptyMap)).cache
        } else {
            glomTrain = glomResults.map(t => (t._1, t._2, t._3, t._4, t._5)).cache
        }
        glomTrain.count
        glomTrain.setName(s"glomTrain $checkpoint")
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

        val featureSize = train.first._2.size
        val nparts = glomTrain.count
        val featuresPerCore = (featureSize.toDouble / nparts).ceil.toInt
        println(s"Feature size: $featureSize")
        println(s"Number of partitions: $nparts")
        println(s"Number of features per partition: $featuresPerCore")

        var curIter = 0
        var terminate = false
        var start = 0
        var gamma = INIT_GAMMA
        while (!terminate && (maxIters == 0 || curIter < maxIters)) {
            val timerStart = System.currentTimeMillis()

            curIter += 1
            println("Node " + localNodes.size)

            // 1. Find a good weak rule
            //    Simulating TMSN using Spark's computation model ==>
            //        Ask workers to scan a batch of examples at a time until one of them find
            //        a valid weak rule (as per early stop rule).
            var resSplit: (Types.ResultType, Double) = (
                (0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0, true), 0.0
            )

            var scanned = 0
            while (gamma > MIN_GAMMA && resSplit._1._1 == 0) {
                scanned = 0
                var nextGamma = 0.0
                setGlomTrain()
                val featureOffset = randomInt(featureSize)
                if (start + scanned > minPartSize) {
                    start = 0
                }

                while (scanned < maxPartSize && resSplit._1._1 == 0) {
                    println(s"Now scan $seqChunks examples from $start, threshold $gamma.")
                    val (glomResults, bestGamma) = learnerFunc(
                        sc, glomTrain, nodes,
                        featureOffset, featuresPerCore,
                        scanned, start, seqChunks, gamma, delta
                    )
                    glomResults.setName(s"glomResults $curIter $scanned")
                    glomResults.cache()
                    checkpoint += 1
                    if (checkpoint % 20 == 0) {
                        if (checkpoint % 100 == 0) {
                            clearCheckpoints()
                        }
                        glomResults.checkpoint()
                        glomTrain.checkpoint()
                        glomResults.count()
                        glomTrain.count()
                        println()
                        println(s"Checkpoint $checkpoint")
                        println()
                    }
                    nextGamma = max(nextGamma, bestGamma)
                    val results = glomResults.map(t => (t._6, t._4)).filter(_._1._1 != 0).cache()
                    if (results.count > 0) {
                        // for simulation: select the earliest stopped one
                        resSplit = results.reduce((a, b) => if (a._1._1 < b._1._1) a else b)
                    } else {
                        setGlomTrain(glomResults)
                    }
                    start = (start + seqChunks) % maxPartSize
                    scanned += seqChunks

                    println("We have " + results.count +
                        s" potential biased rules, best gamma is $bestGamma, " +
                        s"next gamma threshold will be $nextGamma")

                    {
                        // Debug
                        // setGlomTrain(glomResults)
                        // println(glomResults.first._4.keys.toList.take(5).toList)
                        // println(glomResults.first._4.values.toList.head.toList)
                        // println(glomTrain.first._4)
                    }

                    glomResults.unpersist()
                    /*
                    println("Testing progress: most extreme outlier " +
                        safeMaxAbs3(glomTrain.map(t =>
                            safeMaxAbs2(t._4.values.map(safeMaxAbs(scanned)).toIterator)
                        )) + ", threshold " + (scanned))
                    */
                }
                if (resSplit._1._1 == 0) {
                    gamma = nextGamma * 0.85
                    println(s"\nSetting gamma threshold to $gamma\n")
                }
            }

            if (resSplit._1._1 == 0) {
                println("=== !!!  Cannot find a valid weak learner at all.  !!! ===")
                return localNodes
            }

            println(s"Stopped after scanning $scanned examples in (ms) " +
                    (System.currentTimeMillis() - timerStart))

            val ((steps, gamma1, val1, wsum1, wsq1, cnt1, wsum,
                nodeIndex, dimIndex, splitIndex, splitEval), nodeEffectRatio) = resSplit
            seqChunks = ((seqChunks + steps) / 2).ceil.toInt

            val splitVal = 0.5  // TODO: fix this
            val g = gamma1 * wsum / wsum1 - EPS
            var pred = if (val1 > 0) (0.5 * log((1.0 + g) / (1.0 - g)))
                       else           (0.5 * log((1.0 - g) / (1.0 + g)))

            val eff1 = (wsum1 * wsum1) / wsq1 / cnt1
            println(s"$steps steps ($cnt1 hits) achieved score $val1,\n" +
                    s"wsum $wsum1 out of $wsum, wsq $wsq1, effective $eff1\n" +
                    s"g $g, gamma $gamma1\n" +
                    s"Node effective ratio is $nodeEffectRatio")

            println("Depth: " + (localNodes(nodeIndex).depth + 1))
            println(s"Predicts $pred. Father $nodeIndex. " +
                    s"Feature $dimIndex, split at $splitVal, eval $splitEval")

            var redflag = false

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
                    while (i < t._3.size) {
                        val (y, x) = t._2(i)
                        val w = t._3(i)
                        if ((x(dimIndex) <= 0.5) == splitEval && check(x, nodeIndex)) {
                            if (y > 0) {
                                pos += w
                            } else {
                                neg += w
                            }
                        }
                        i+= 1
                    }
                    (pos, neg)
                }).reduce((a, b) =>
                    (a._1 + b._1, a._2 + b._2)
                )
                val actualPred = 0.5 * log(pos / neg)
                println("Actual prediction should be " + 0.5 * log(pos / neg) + " (gamma=" +
                        abs(pos - neg) / (pos + neg) + ")")
                if (actualPred > 0 && pred < 0 || actualPred < 0 && pred > 0 || abs(actualPred) < abs(pred)) {
                    println("=== ERROR: overweightted/overfitted tree node detected ===")
                    pred = actualPred
                }
            }

            if (!redflag) {

            // add the new node to the nodes list
            val newNodeId = nodes.size
            val newNode = SplitterNode(newNodeId, nodeIndex, localNodes(nodeIndex).depth + 1,
                                       (dimIndex, splitVal, splitEval))
            newNode.setPredict(pred)
            localNodes(nodeIndex).addChild(localNodes.size)
            val brNewNode = sc.broadcast(newNode)
            nodes :+= brNewNode
            localNodes :+= newNode

            val toDestroy = glomTrain
            glomTrain = updateFunc(glomTrain, nodes).cache()
            glomTrain.setName("glomTrain after updateFunc")
            val effectCounts = glomTrain.map(_._4).collect.sorted
            val numParts = effectCounts.size
            val numValidParts = effectCounts.count(_ > 0.1)
            println(s"Effective parts: $numValidParts out of $numParts")
            println((effectCounts.slice(0, 5) ++
                effectCounts.slice(effectCounts.size - 5, effectCounts.size)).toList)
            toDestroy.unpersist()

            if (curIter % printStatsInterval == 0) {
                SplitterNode.save(localNodes, modelWritePath)
                println("Wrote model to disk at iteration " + curIter)

                val timerPrint = System.currentTimeMillis()
                val (curTrainAvgScore, curTestAvgScore) = Utils.printStats(
                    train, glomTrain, test, testRef, localNodes, curIter
                )
                println("printStats took (ms) " + (System.currentTimeMillis() - timerPrint))
            }

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
