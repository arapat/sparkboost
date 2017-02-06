package sparkboost

import collection.mutable.ListBuffer
import math.exp
import math.log
import util.Random.{nextDouble => rand}

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import sparkboost.utils.Comparison

object Controller extends Comparison {
    type RDDType = RDD[(List[Instance], Int, List[Double])]
    type LossFunc = (Double, Double, Double, Double, Double) => Double
    type LearnerObj = (Int, Boolean, Condition)
    type LearnerFunc = (RDDType, ListBuffer[SplitterNode], LossFunc, Int) => LearnerObj
    type UpdateFunc = (RDDType, SplitterNode) => RDDType

    def printStats(train: RDDType, test: RDD[Array[Instance]], nodes: List[SplitterNode],
                   iter: Int) {
        val trainError = train.map(_._1.filter(t => t.w >= 1.0).size).reduce(_ + _)
        val trainTotal = train.map(_._1.size).reduce(_ + _)
        val trainErrorRate = trainError.toDouble / trainTotal
        val trainWSum = train.map(_._1.map(t => t.w).reduce(_ + _)).reduce(_ + _)
        val trainWSqSum = train.map(_._1.map(t => t.w * t.w).reduce(_ + _)).reduce(_ + _)
        val effectCnt = (trainWSum.toDouble * trainWSum / trainWSqSum) / trainTotal
        val testError = test.map(
                            _.filter(t => SplitterNode.getScore(0, nodes, t) * t.y <= 1e-8)
                             .size
                        ).reduce(_ + _)
        val testTotal = test.map(_.size).reduce(_ + _)
        val testErrorRate = testError.toDouble / testTotal
        println("Iteration " + iter)
        println("Training error is " + trainErrorRate)
        println("Test error is " + testErrorRate)
        println("Effective count ratio is " + effectCnt)
    }

    def sample(data: RDDType,
               fraction: Double,
               nodes: List[SplitterNode]): RDDType = {
        println("Resampling...")
        // TODO: extends the cost functions to other forms
        val sampleData = data.map(datum => {
            val array = datum._1
            val sampleList = ListBuffer[Instance]()
            val size = (array.size * fraction).ceil.toInt
            val weights = array.map(t => exp(-t.y * SplitterNode.getScore(0, nodes, t)))
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
        }).cache()
        println("Resampling done. Sample size: " + sampleData.map(_._1.size).reduce(_ + _))
        sampleData
    }

    def runADTree(instances: RDD[Instance],
                  test: RDD[Instance],
                  learnerFunc: LearnerFunc,
                  updateFunc: UpdateFunc,
                  lossFunc: LossFunc,
                  sliceFrac: Double,
                  sampleFrac: Double,
                  K: Int, T: Int) = {
        def preprocess(data: (Array[Instance], Long)) = {
            val insts = data._1
            val index = data._2.toInt
            val sortedInsts = insts.toList.sortWith(_.X(index) < _.X(index))

            // Generate the slices
            val slices = ListBuffer(Double.MinValue)
            val sliceSize = (insts.size * sliceFrac).floor.toInt
            var lastValue = Double.MinValue
            var lastPos = 0
            var curPos = 0
            for (t <- sortedInsts) {
                if (curPos - lastPos >= sliceSize && compare(lastValue, t.X(index)) != 0) {
                    lastPos = curPos
                    slices.append(0.5 * (lastValue + t.X(index)))
                }
                lastValue = t.X(index)
                curPos = curPos + 1
            }
            slices.append(Double.MaxValue)
            (sortedInsts, index, slices.toList)
        }

        // assure the feature size is equal to the partition size
        require(instances.partitions.size == instances.first.X.size)

        // Glom data
        val glomTrain = instances.glom().zipWithIndex().map(preprocess).cache()
        val glomTest = test.coalesce(10).glom().cache()

        // Set up the root of the ADTree
        val posCount = instances filter {t => t.y > 0} count
        val negCount = instances.count - posCount
        val predVal = 0.5 * log(posCount.toDouble / negCount)
        val rootNode = SplitterNode(0, new TrueCondition(), -1, true)
        rootNode.setPredict(predVal, 0.0)
        val nodes = ListBuffer(rootNode)

        // Iteratively grow the ADTree
        for (batch <- 0 until T / K) {
            // Set up instances RDD
            var data = sample(glomTrain, sampleFrac, nodes.toList)

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
                val minVal = predicts.values.filter(compare(_) > 0).min * 0.001
                val leftPos = predicts.getOrElse((1, 1), minVal)
                val leftNeg = predicts.getOrElse((1, -1), minVal)
                val rightPos = predicts.getOrElse((-1, 1), minVal)
                val rightNeg = predicts.getOrElse((-1, -1), minVal)
                val leftPred = 0.5 * log(leftPos.toDouble / leftNeg)
                val rightPred = 0.5 * log(rightPos.toDouble / rightNeg)
                newNode.setPredict(leftPred, rightPred)

                // add the new node to the nodes list
                nodes(prtNodeIndex).addChild(onLeft, nodes.size)
                nodes.append(newNode)

                // adjust the weights of the instances
                // TODO: why caching will slow the program down?
                data = updateFunc(data, newNode).persist(StorageLevel.MEMORY_ONLY)
                /*
                if (iteration % 25 == 0) {
                    data.checkpoint()
                }
                */
                printStats(data, glomTest, nodes.toList, batch * K + iteration)
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
                  LossFunc.lossfunc, sliceFrac, sampleFrac, K, T)
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
                  LossFunc.lossfunc, sliceFrac, sampleFrac, K, T)
    }

    /*
    def runADTreeWithBulkAdaboost(instances: RDD[Instance], T: Int) = {
        runADTree(instances, Learner.bulkGreedySplit, UpdateFunc.adaboostUpdate,
                  LossFunc.lossfunc, T)
    }
    */
}
