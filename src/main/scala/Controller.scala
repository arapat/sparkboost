package sparkboost

import collection.mutable.ListBuffer
import math.log

import org.apache.spark.rdd.RDD

object Controller {
    type Instance = (Int, Vector[Double], Double)
    type LossFunc = (Double, Double, Double, Double, Double) => Double
    type LearnerObj = (SplitterNode, Boolean, Condition)
    type LearnerFunc = (RDD[Instance], ListBuffer[SplitterNode], LossFunc, Boolean, Int) => LearnerObj
    type UpdateFunc = (RDD[Instance], SplitterNode) => RDD[Instance]

    def runADTree(instances: RDD[Instance],
                  learnerFunc: LearnerFunc, updateFunc: UpdateFunc, lossFunc: LossFunc,
                  T: Int, repartition: Boolean) = {
        // Set up the root of the ADTree
        val posCount = instances filter {t => t._1 > 0} count
        val negCount = instances.count - posCount
        val predVal = 0.5 * log(posCount.toDouble / negCount)
        val rootNode = SplitterNode(0, new TrueCondition(),
                                    {_ : Vector[Double] => true}, true)
        rootNode.setPredict(predVal, 0.0)

        // Set up instances RDD
        var data = updateFunc(instances, rootNode).cache()

        // Iteratively grow the ADTree
        var nodes = ListBuffer(rootNode)
        for (iteration <- 1 until T) {
            val bestSplit = learnerFunc(data, nodes, lossFunc, false, 0)
            val prtNode = bestSplit._1
            val onLeft = bestSplit._2
            val condition = bestSplit._3
            val newNode = SplitterNode(
                    nodes.size, condition,
                    {(t: Vector[Double]) => !((prtNode.check(t, false) > 0) ^ onLeft)},
                    onLeft
            )

            // compute the predictions of the new node
            val predicts = (
                instances.map {
                    t: Instance => ((newNode.check(t._2, false), t._1), t._3)
                }.filter {
                    t => t._1._1 != 0
                }.reduceByKey {
                    (a: Double, b: Double) => a + b
                }.collectAsMap()
            )
            val minVal = predicts.size * 0.001
            val leftPos = predicts.getOrElse((1, 1), minVal)
            val leftNeg = predicts.getOrElse((1, -1), minVal)
            val rightPos = predicts.getOrElse((-1, 1), minVal)
            val rightNeg = predicts.getOrElse((-1, -1), minVal)
            val leftPred = 0.5 * log(leftPos.toDouble / leftNeg)
            val rightPred = 0.5 * log(rightPos.toDouble / rightNeg)
            newNode.setPredict(leftPred, rightPred)

            // add the new node to the nodes list
            nodes(prtNode.index).addChild(onLeft, nodes.size)
            nodes += newNode

            // adjust the weights of the instances
            data = updateFunc(data, newNode).cache()
            if (iteration % 10 == 0) {
                data.checkpoint()
            }
        }
        nodes
    }

    def runADTreeWithAdaBoost(instances: RDD[Instance], T: Int, repartition: Boolean) = {
        runADTree(instances, Learner.partitionedGreedySplit, UpdateFunc.adaboostUpdate,
                  LossFunc.lossfunc, T, repartition)
    }

    def runADTreeWithLogitBoost(instances: RDD[Instance], T: Int, repartition: Boolean) = {
        runADTree(instances, Learner.partitionedGreedySplit, UpdateFunc.logitboostUpdate,
                  LossFunc.lossfunc, T, repartition)
    }
}
