package sparkboost

import collection.mutable.ListBuffer
import math.log
import util.Random.{nextDouble => rand}

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import sparkboost.utils.Comparison

object Controller extends Comparison {
    type RDDType = RDD[(List[Instance], Int, List[Double])]
    type LossFunc = (Double, Double, Double, Double, Double) => Double
    type LearnerObj = (Int, Boolean, Condition)
    type LearnerFunc = (RDDType, ListBuffer[SplitterNode], LossFunc, Int) => LearnerObj
    type UpdateFunc = (RDDType, SplitterNode) => RDDType
    type WeightFunc = (Int, Double, Double) => Double

    def printStats(train: RDDType, test: RDD[Array[Instance]], nodes: List[SplitterNode],
                   iter: Int) {
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
        val testPredictAndLabels = test.flatMap(_.map(t => {
            val predict = SplitterNode.getScore(0, nodes, t)
            (predict.toDouble, t.y.toDouble)
        }))
        val trainWSum = train.map(_._1.map(t => t.w).reduce(_ + _)).reduce(_ + _)
        val trainWSqSum = train.map(_._1.map(t => t.w * t.w).reduce(_ + _)).reduce(_ + _)
        val effectCnt = (trainWSum.toDouble * trainWSum / trainWSqSum) / trainCount

        // Instantiate metrics object
        val trainMetrics = new BinaryClassificationMetrics(trainPredictionAndLabels)
        val train_auPRC = trainMetrics.areaUnderPR + adjust(trainMetrics.pr.take(2))
        val testMetrics = new BinaryClassificationMetrics(testPredictAndLabels)
        val test_auPRC = testMetrics.areaUnderPR + adjust(testMetrics.pr.take(2))

        println("(Training) auPRC = " + train_auPRC)
        println("(Test) auPRC = " + test_auPRC)
        println("Effective count ratio is " + effectCnt)
    }

    def sample(data: RDDType,
               fraction: Double,
               nodes: List[SplitterNode],
               wfunc: (Int, Double, Double) => Double): RDDType = {
        println("Resampling...")
        val sampleData = data.map(datum => {
            val array = datum._1
            val sampleList = ListBuffer[Instance]()
            val size = (array.size * fraction).ceil.toInt
            val weights = array.map(t => wfunc(t.y, 1.0, SplitterNode.getScore(0, nodes, t)))
            val weightSum = weights.reduce(_ + _)
            val segsize = weightSum.toDouble / size

            var curWeight = rand() * segsize // first sample point
            var accumWeight = 0.0
            for (iw <- array.zip(weights)) {
                while (accumWeight <= curWeight && curWeight < accumWeight + iw._2) {
                    sampleList.append(iw._1)
                    curWeight += segsize
                }
                accumWeight += iw._2
            }
            for (s <- sampleList) {
                s.setWeight(1.0)
                s.setScores(nodes)
            }
            (sampleList.toList, datum._2, datum._3)
        })
        sampleData.checkpoint()
        println("Resampling done. Sample size: " + sampleData.map(_._1.size).reduce(_ + _))
        sampleData
    }

    def runADTree(train: RDD[Instance],
                  test: RDD[Instance],
                  learnerFunc: LearnerFunc,
                  updateFunc: UpdateFunc,
                  lossFunc: LossFunc,
                  weightFunc: WeightFunc,
                  sliceFrac: Double,
                  sampleFrac: Double,
                  K: Int, T: Int): ListBuffer[SplitterNode] = {
        def safeLogRatio(a: Double, b: Double) = {
            if (compare(a) == 0 && compare(b) == 0) {
                0.0
            } else {
                val ratio = math.min(10.0, math.max(a / b, 0.1))
                log(ratio)
            }
        }

        def preprocess(featureSize: Int)(partIndex: Int, data: Iterator[Instance]) = {
            val index = partIndex % featureSize
            val sortedInsts = data.toList.sortWith(_.X(index) < _.X(index))

            // Generate the slices
            val sliceSize = (sortedInsts.size * sliceFrac).floor.toInt
            val slices =
                (sliceSize until sortedInsts.size by sliceSize).map(
                    idx => 0.5 * (sortedInsts(idx - 1).X(index) + sortedInsts(idx).X(index))
                ).distinct.toList :+ Double.MaxValue
            Iterator((sortedInsts, index, slices))
        }

        def getMetaInfo(insts: List[Instance]) = {
            val posInsts = insts.count(_.y > 0)
            (insts.size, posInsts, insts.size - posInsts)
        }

        // assure the feature size is equal to the partition size
        val featureSize = train.first.X.size
        require(train.partitions.size >= featureSize)

        // Glom data
        val glomTrain = train.mapPartitionsWithIndex(preprocess(featureSize))
                             .cache()
        val glomTest = test.coalesce(10).glom().cache()

        // print meta info about partitions
        val metas = glomTrain.map(_._1).map(getMetaInfo).collect
        println("Number of partitions: " + metas.size)
        println(metas.reduce((a, b) => if (a._2 < b._2) a else b))

        // Set up the root of the ADTree
        val posCount = glomTrain map {t => t._1.count(_.y > 0)} reduce(_ + _)
        val negCount = glomTrain.map(_._1.size).reduce(_ + _) - posCount
        println(s"Positive examples: $posCount")
        println(s"Negative examples: $negCount")
        println()

        val predVal = 0.5 * log(posCount.toDouble / negCount)
        val rootNode = SplitterNode(0, new TrueCondition(), -1, true)
        rootNode.setPredict(predVal, 0.0)
        val nodes = ListBuffer(rootNode)
        println(s"Predict ($predVal, 0.0)")
        printStats(sample(glomTrain, sampleFrac, nodes.toList, weightFunc),
                   glomTest, nodes.toList, 0)
        println()

        // Iteratively grow the ADTree
        for (batch <- 0 until T / K) {
            // Set up instances RDD
            var data = sample(glomTrain, sampleFrac, nodes.toList, weightFunc)
            // println("New positive sample weight:")
            // data.map(_._1.filter(_.y > 0).map(_.w)).reduce(_ ::: _).foreach(t => print("%.2f, ".format(t)))
            // println("New negative sample weight: " + data.first._1.filter(_.y < 0).head.w)

            for (iteration <- 1 to K) {
                val bestSplit = learnerFunc(data, nodes, lossFunc, 0)
                val prtNodeIndex = bestSplit._1
                val onLeft = bestSplit._2
                val condition = bestSplit._3
                val newNode = SplitterNode(nodes.size, condition, prtNodeIndex, onLeft)

                // compute the predictions of the new node
                val predicts = (
                    data.flatMap(
                        _._1
                    ).map {
                        t: Instance => ((newNode.check(t), t.y), t.w)
                    }.filter {
                        t => t._1._1 != 0
                    }.reduceByKey {
                        (a: Double, b: Double) => a + b
                    }.collectAsMap()
                )
                println("predicts:" + predicts)
                val leftPos = predicts.getOrElse((1, 1), 0.0)
                val leftNeg = predicts.getOrElse((1, -1), 0.0)
                val rightPos = predicts.getOrElse((-1, 1), 0.0)
                val rightNeg = predicts.getOrElse((-1, -1), 0.0)
                val leftPred = 0.5 * safeLogRatio(leftPos.toDouble, leftNeg.toDouble)
                val rightPred = 0.5 * safeLogRatio(rightPos.toDouble, rightNeg.toDouble)
                newNode.setPredict(leftPred, rightPred)
                println(s"Predicts ($leftPred, $rightPred)")

                // add the new node to the nodes list
                nodes(prtNodeIndex).addChild(onLeft, nodes.size)
                nodes.append(newNode)

                // adjust the weights of the instances
                // TODO: why caching will slow the program down?
                // println("(before) Positive sample weight:")
                // data.map(_._1.filter(_.y > 0).map(_.w)).reduce(_ ::: _).foreach(t => print("%.2f, ".format(t)))
                data = updateFunc(data, newNode).persist(StorageLevel.MEMORY_ONLY)
                // println("(after) Positive sample weight:")
                // println(data.map(_._1.filter(_.y > 0)).reduce(_ ::: _).foreach(t => println("%.2f ".format(t.w) + t.X)))
                // return nodes
                // println("Negative sample weight: " + data.first._1.filter(_.y < 0).head.w)
                /*
                if (iteration % 25 == 0) {
                    data.checkpoint()
                }
                */
                printStats(data, glomTest, nodes.toList, batch * K + iteration)
                println
            }
        }
        nodes
    }

    def runADTreeWithAdaBoost(instances: RDD[Instance], test: RDD[Instance], sliceFrac: Double,
                              sampleFrac: Double, K: Int, T: Int, repartition: Boolean) = {
        val data =
            if (repartition) {
                val featureSize = instances.first.X.size
                instances.repartition(featureSize)
            } else {
                instances
            }
        runADTree(data, test, Learner.partitionedGreedySplit, UpdateFunc.adaboostUpdate,
                  LossFunc.lossfunc, UpdateFunc.adaboostUpdateFunc, sliceFrac, sampleFrac, K, T)
    }

    def runADTreeWithLogitBoost(instances: RDD[Instance], test: RDD[Instance], sliceFrac: Double,
                                sampleFrac: Double, K: Int, T: Int, repartition: Boolean) = {
        val data =
            if (repartition) {
                val featureSize = instances.first.X.size
                instances.repartition(featureSize)
            } else {
                instances
            }
        runADTree(instances, test, Learner.partitionedGreedySplit, UpdateFunc.logitboostUpdate,
                  LossFunc.lossfunc, UpdateFunc.logitboostUpdateFunc, sliceFrac, sampleFrac, K, T)
    }

    /*
    def runADTreeWithBulkAdaboost(instances: RDD[Instance], T: Int) = {
        runADTree(instances, Learner.bulkGreedySplit, UpdateFunc.adaboostUpdate,
                  LossFunc.lossfunc, T)
    }
    */
}
