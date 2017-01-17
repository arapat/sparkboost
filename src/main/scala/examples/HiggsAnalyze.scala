package sparkboost.examples

import collection.mutable.ListBuffer

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import sparkboost._
import sparkboost.utils.Comparison

object HiggsAdaboostAnalyze extends Comparison {
    /*
    args(0) - master node URL
    args(1) - file path to the training data
    args(2) - file path to the testing data
    args(3) - File path where the model is saved
    args(4) - which algorithm to use - 1 for AdaBoost, 2 for LogitBoost
    */
    def main(args: Array[String]) {
        def getNewPredict(inst: Instance, node: SplitterNode) = {
            val s = inst.scores.last
            inst.y * (s match {
                case 1  => node.leftPredict
                case -1 => node.rightPredict
                case _  => 0.0
            })
        }

        if (args.size != 5) {
            println(
                "Please provide four arguments: master url, training data path, " +
                "testing data path, model file path, algorithm ID."
            )
            return
        }

        val conf = new SparkConf().setMaster(args(0))
        val sc = new SparkContext(conf)
        sc.setCheckpointDir("checkpoints/")
        var train = sc.textFile(args(1))
                      .map {line => line.split(",").map(_.trim.toDouble)}
                      .map {t => Instance((t.head + t.head - 1.0).toInt, t.tail.toVector)}
                      .cache()
        val trainSize = train.count
        var test = sc.textFile(args(2))
                     .map {line => line.split(",").map(_.trim.toDouble)}
                     .map {t => Instance((t.head + t.head - 1.0).toInt, t.tail.toVector)}
                     .cache()
        val testSize = test.count

        val updateFunc = if (args(4).toInt == 1) UpdateFunc.adaboostUpdate _
                         else                    UpdateFunc.logitboostUpdate _
        val nodes = SplitterNode.load(args(3))
        var trainPredicts = (0 until train.count.toInt).map(t => 0.0).toList
        var testPredicts = (0 until test.count.toInt).map(t => 0.0).toList
        var trainError = ListBuffer[Double]()
        var testError = ListBuffer[Double]()
        var ec = ListBuffer(train.count.toDouble)
        var iters = 0
        for (node <- nodes) {
            iters = iters + 1
            train = updateFunc(train, node).cache()
            test = updateFunc(test, node).cache()
            if (iters % 25 == 0) {
                train.checkpoint()
                test.checkpoint()
            }

            val newTrainPredicts = train.map(t => getNewPredict(t, node)).collect
            trainPredicts = trainPredicts.zip(newTrainPredicts).map(t => t._1 + t._2)
            val newTestPredicts = test.map(t => getNewPredict(t, node)).collect
            testPredicts = testPredicts.zip(newTestPredicts).map(t => t._1 + t._2)

            trainError += trainPredicts.filter(t => compare(t) <= 0).size.toDouble / trainSize
            testError += testPredicts.filter(t => compare(t) <= 0).size.toDouble / testSize
            val weights = train.map(_.w)
            val wsum: Double = weights.reduce(_ + _)
            val wsq: Double = weights.map {x => x * x}
                                     .reduce(_ + _)
            ec += ((wsum * wsum).toDouble / wsq)
        }

        def join(array : ListBuffer[Double]) = {
            array.tail.foldLeft(array.head.toString)((s, k) => s + ", " + k)
        }
        println("Training error:")
        println(join(trainError))
        println("Test error:")
        println(join(testError))
        println("Effective count:")
        println(join(ec))
    }
}
