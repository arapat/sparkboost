package sparkboost

import math.log
import util.Random.{nextDouble => rand}

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.SparseVector

import java.io._

import sparkboost.utils.Comparison

object Controller extends Comparison {
    val BINSIZE = 1
    type RDDType = RDD[Instances]
    type TestRDDType = RDD[Array[(Int, SparseVector)]]
    type LossFunc = (Double, Double, Double, Double, Double) => Double
    type LearnerObj = (Int, Boolean, Int, Double, Double, Double)
    type LearnerFunc = (RDDType, Array[SplitterNode], LossFunc, Int, Int) => LearnerObj
    type UpdateFunc = (RDDType, SplitterNode) => RDDType
    type WeightFunc = (Int, Double, Double) => Double

    def printStats(train: RDDType, test: TestRDDType, nodes: Array[SplitterNode],
                   iter: Int) = {
        // manual fix the auPRC computation bug in MLlib
        def adjust(points: Array[(Double, Double)]) = {
            require(points.length == 2)
            require(points.head == (0.0, 1.0))
            val y = points.last
            y._1 * (y._2 - 1.0) / 2.0
        }

        val trainPredictionAndLabels = train.flatMap(_._1.map(t => {
            val predict = SplitterNode.getScore(0, nodes, t)
            (predict.toDouble, t.y.toDouble)
        })).cache()
        val trainCount = trainPredictionAndLabels.count()
        val posTrainCount = trainPredictionAndLabels.filter(_._2 > 0).count()
        val negTrainCount = trainPredictionAndLabels.filter(_._2 < 0).count()
        val testPredictAndLabels = test.flatMap(_.map(t => {
            val predict = SplitterNode.getScore(0, nodes, t)
            (predict.toDouble, t.y.toDouble)
        }))
        val trainWSum = train.map(_._1.map(t => t.w).reduce(_ + _)).reduce(_ + _)
        val trainWSqSum = train.map(_._1.map(t => t.w * t.w).reduce(_ + _)).reduce(_ + _)
        val effectCnt = (trainWSum.toDouble * trainWSum / trainWSqSum) / trainCount
        val posTrainWSum = train.map(_._1.filter(t => t.y > 0).map(t => t.w).reduce(_ + _)).reduce(_ + _)
        val posTrainWSqSum = train.map(_._1.filter(t => t.y > 0).map(t => t.w * t.w).reduce(_ + _)).reduce(_ + _)
        val posEffectCnt = (posTrainWSum.toDouble * posTrainWSum / posTrainWSqSum) / posTrainCount
        val negTrainWSum = train.map(_._1.filter(t => t.y < 0).map(t => t.w).reduce(_ + _)).reduce(_ + _)
        val negTrainWSqSum = train.map(_._1.filter(t => t.y < 0).map(t => t.w * t.w).reduce(_ + _)).reduce(_ + _)
        val negEffectCnt = (negTrainWSum.toDouble * negTrainWSum / negTrainWSqSum) / negTrainCount

        // Instantiate metrics object
        val trainMetrics = new BinaryClassificationMetrics(trainPredictionAndLabels)
        val train_auPRC = trainMetrics.areaUnderPR + adjust(trainMetrics.pr.take(2))
        val testMetrics = new BinaryClassificationMetrics(testPredictAndLabels)
        val test_auPRC = testMetrics.areaUnderPR + adjust(testMetrics.pr.take(2))

        println("(Training) auPRC = " + train_auPRC)
        // println("Training PR = " + trainMetrics.pr.collect)
        println("(Test) auPRC = " + test_auPRC)
        println("Effective count = " + effectCnt)
        println("Positive effective count = " + posEffectCnt)
        println("Negative effective count = " + negEffectCnt)
        effectCnt
    }

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

    def runADTree(glomTrain: RDDType,
                  glomTest: TestRDDType,
                  learnerFunc: LearnerFunc,
                  updateFunc: UpdateFunc,
                  lossFunc: LossFunc,
                  weightFunc: WeightFunc,
                  sliceFrac: Double,
                  sampleFrac: Double,
                  T: Int, maxDepth: Int,
                  baseNodes: Array[SplitterNode]): Array[SplitterNode] = {
        def safeLogRatio(a: Double, b: Double) = {
            if (compare(a) == 0 && compare(b) == 0) {
                0.0
            } else {
                val ratio = math.min(10.0, math.max(a / b, 0.1))
                log(ratio)
            }
        }

        def getMetaInfo(insts: List[Instance]) = {
            val posInsts = insts.count(_.y > 0)
            (insts.size, posInsts, insts.size - posInsts)
        }

        // print meta info about partitions
        val metas = glomTrain.map(_._1).map(getMetaInfo).collect
        println("Number of partitions: " + metas.size)
        println(metas.reduce((a, b) => if (a._2 < b._2) a else b))

        // Set up the root of the ADTree
        val posCount = glomTrain map {t => t._1.count(_.y > 0)} reduce(_ + _)
        val negCount = glomTrain.map(_._1.size).reduce(_ + _) - posCount
        println(s"Positive examples: $posCount")
        println(s"Negative examples: $negCount")
        val testPosCount = glomTest.map(_.count(_.y > 0)).reduce(_ + _)
        val testNegCount = glomTest.map(_.count(_.y < 0)).reduce(_ + _)
        println(s"Test positive examples: $testPosCount")
        println(s"Test negative examples: $testNegCount")

        var data = glomTrain
        var nodes =
            if (baseNodes == null) {
                val predVal = 0.5 * log(posCount.toDouble / negCount)
                val rootNode = SplitterNode(0, -1, 0, -1, true)
                rootNode.setPredict(predVal, 0.0)
                println(s"Predict ($predVal, 0.0)")
                data = updateFunc(glomTrain, rootNode).persist(StorageLevel.MEMORY_ONLY)
                data.count()
                glomTrain.unpersist()
                Array(rootNode)
            } else {
                baseNodes
            }

        printStats(data.filter(_._2 < BINSIZE), glomTest, nodes, 0)
        println()

        var iteration = 0
        while (iteration < T) {
            iteration = iteration + 1

            val bestSplit = learnerFunc(data, nodes, lossFunc, maxDepth, 0)
            val prtNodeIndex = bestSplit._1
            val onLeft = bestSplit._2
            val splitIndex = bestSplit._3
            val splitVal = bestSplit._4
            val newNode = SplitterNode(nodes.size, splitIndex, splitVal, prtNodeIndex, onLeft)

            // compute the predictions of the new node
            val predicts = (
                data.filter(
                    _._2 < BINSIZE
                ).flatMap(
                    _._1
                ).map {
                    t: Instance => ((newNode.check(t), t.y), t.w)
                }.reduceByKey {
                    (a: Double, b: Double) => a + b
                }.collectAsMap()
            )
            val predictsCount = (
                data.filter(
                    _._2 < BINSIZE
                ).flatMap(
                    _._1
                ).map {
                    t: Instance => ((newNode.check(t), t.y), 1)
                }.reduceByKey {
                    (a: Int, b: Int) => a + b
                }.collectAsMap()
            )
            val verify = {
                data.filter(
                    _._2 == splitIndex
                ).flatMap(
                    _._1
                ).map {
                    t: Instance => ((newNode.check(t), t.y), 1)
                }.reduceByKey {
                    (a: Int, b: Int) => a + b
                }.collectAsMap()
            }
            println("predicts: " + predicts)
            println("counts: " + predictsCount)
            println("verify: " + verify)
            val leftPos = predicts.getOrElse((1, 1), 0.0)
            val leftNeg = predicts.getOrElse((1, -1), 0.0)
            val rightPos = predicts.getOrElse((-1, 1), 0.0)
            val rightNeg = predicts.getOrElse((-1, -1), 0.0)
            val leftPred = 0.5 * safeLogRatio(leftPos.toDouble, leftNeg.toDouble)
            val rightPred = 0.5 * safeLogRatio(rightPos.toDouble, rightNeg.toDouble)
            /*
            val leftPred = bestSplit._4._1
            val rightPred = bestSplit._4._2
            */
            newNode.setPredict(leftPred, rightPred)
            println(s"Predicts ($leftPred, $rightPred) Father $prtNodeIndex")

            // add the new node to the nodes list
            nodes(prtNodeIndex).addChild(onLeft, nodes.size)
            nodes :+= newNode

            val oldData = data
            data = updateFunc(data, newNode).persist(StorageLevel.MEMORY_ONLY)  // _SER)
            data.count()
            oldData.unpersist()

            val effcnt = printStats(
                data.filter(_._2 < BINSIZE), glomTest, nodes, iteration
            )

            println
        }

        // print visualization meta data for JBoost
        val posTrain = data.flatMap(_._1).filter(_.y > 0).takeSample(true, 3000)
        val negTrain = data.flatMap(_._1).filter(_.y < 0).takeSample(true, 3000)
        val posTest = glomTest.flatMap(t => t).filter(_.y > 0).takeSample(true, 3000)
        val negTest = glomTest.flatMap(t => t).filter(_.y < 0).takeSample(true, 3000)
        val esize = 6000

        val trainFile = new File("trial0.train.boosting.info")
        val trainWrite = new BufferedWriter(new FileWriter(trainFile))
        val testFile = new File("trial0.test.boosting.info")
        val testWrite = new BufferedWriter(new FileWriter(testFile))

        for (i <- 1 to nodes.size) {
            trainWrite.write(s"iteration=$i : elements=$esize : boosting_params=None (jboost.booster.AdaBoost):\n")
            testWrite.write(s"iteration=$i : elements=$esize : boosting_params=None (jboost.booster.AdaBoost):\n")
            var id = 0
            for (t <- posTrain) {
                val score = SplitterNode.getScore(0, nodes, t, i)
                trainWrite.write(s"$id : $score : $score : 1 : \n")
                id = id + 1
            }
            for (t <- negTrain) {
                val score = SplitterNode.getScore(0, nodes, t, i)
                val negscore = -score
                trainWrite.write(s"$id : $negscore : $score : -1 : \n")
                id = id + 1
            }
            id = 0
            for (t <- posTest) {
                val score = SplitterNode.getScore(0, nodes, t, i)
                testWrite.write(s"$id : $score : $score : 1 : \n")
                id = id + 1
            }
            for (t <- negTest) {
                val score = SplitterNode.getScore(0, nodes, t, i)
                val negscore = -score
                testWrite.write(s"$id : $negscore : $score : -1 : \n")
                id = id + 1
            }
        }

        trainWrite.close()
        testWrite.close()
        nodes
    }

    def runADTreeWithAdaBoost(instances: RDDType, test: TestRDDType, sliceFrac: Double,
                              sampleFrac: Double, T: Int, maxDepth: Int, baseNodes: Array[SplitterNode]) = {
        runADTree(instances, test, Learner.partitionedGreedySplit, UpdateFunc.adaboostUpdate,
                  LossFunc.lossfunc, UpdateFunc.adaboostUpdateFunc, sliceFrac, sampleFrac, T,
                  maxDepth, baseNodes)
    }

    def runADTreeWithLogitBoost(instances: RDDType, test: TestRDDType, sliceFrac: Double,
                                sampleFrac: Double, T: Int, maxDepth: Int, baseNodes: Array[SplitterNode]) = {
        runADTree(instances, test, Learner.partitionedGreedySplit, UpdateFunc.logitboostUpdate,
                  LossFunc.lossfunc, UpdateFunc.logitboostUpdateFunc, sliceFrac, sampleFrac, T,
                  maxDepth, baseNodes)
    }

    /*
    def runADTreeWithBulkAdaboost(instances: RDD[Instance], T: Int) = {
        runADTree(instances, Learner.bulkGreedySplit, UpdateFunc.adaboostUpdate,
                  LossFunc.lossfunc, T)
    }
    */
}
