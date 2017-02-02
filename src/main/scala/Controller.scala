package sparkboost

import collection.mutable.ListBuffer
import math.log

import org.apache.spark.rdd.RDD

import sparkboost.utils.Comparison

object Controller extends Comparison {
    type RDDType = RDD[(List[Instance], Int, List[Double])]
    type LossFunc = (Double, Double, Double, Double, Double) => Double
    type LearnerObj = (Int, Boolean, Condition)
    type LearnerFunc = (RDDType, ListBuffer[SplitterNode], LossFunc, Int) => LearnerObj
    type UpdateFunc = (RDDType, SplitterNode) => RDDType

    def runADTree(instances: RDD[Instance],
                  learnerFunc: LearnerFunc,
                  updateFunc: UpdateFunc,
                  lossFunc: LossFunc,
                  sliceFrac: Double,
                  T: Int) = {
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

        // Set up the root of the ADTree
        val posCount = instances filter {t => t.y > 0} count
        val negCount = instances.count - posCount
        val predVal = 0.5 * log(posCount.toDouble / negCount)
        val rootNode = SplitterNode(0, new TrueCondition(), -1, true)
        rootNode.setPredict(predVal, 0.0)

        // Set up instances RDD
        val instsGroup = instances.glom().zipWithIndex().map(preprocess)
        var data = updateFunc(instsGroup, rootNode)

        // Iteratively grow the ADTree
        val nodes = ListBuffer(rootNode)
        for (iteration <- 1 until T) {
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
            data = updateFunc(data, newNode)
            if (iteration % 25 == 0) {
                data.checkpoint()
            }
        }
        nodes
    }

    def runADTreeWithAdaBoost(instances: RDD[Instance], sliceFrac: Double, T: Int,
                              repartition: Boolean) = {
        val data =
            if (repartition) {
                val featureSize = instances.first.X.size
                instances.repartition(featureSize)
            } else {
                instances
            }
        runADTree(data, Learner.partitionedGreedySplit, UpdateFunc.adaboostUpdate,
                  LossFunc.lossfunc, sliceFrac, T)
    }

    def runADTreeWithLogitBoost(instances: RDD[Instance], sliceFrac: Double, T: Int,
                                repartition: Boolean) = {
        val data =
            if (repartition) {
                val featureSize = instances.first.X.size
                instances.repartition(featureSize)
            } else {
                instances
            }
        runADTree(instances, Learner.partitionedGreedySplit, UpdateFunc.logitboostUpdate,
                  LossFunc.lossfunc, sliceFrac, T)
    }

    /*
    def runADTreeWithBulkAdaboost(instances: RDD[Instance], T: Int) = {
        runADTree(instances, Learner.bulkGreedySplit, UpdateFunc.adaboostUpdate,
                  LossFunc.lossfunc, T)
    }
    */
}
