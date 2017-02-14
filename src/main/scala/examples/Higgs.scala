package sparkboost.examples

import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import sparkboost._

object Higgs {
    /*
    args(0) - file path to the training data
    args(1) - file path to the test data
    args(2) - fraction for sampling
    args(3) - number of batches
    args(4) - number of iterations
    args(5) - algorithm selection
        1 -> AdaBoost partitioned
        2 -> AdaBoost non-partitioned
        3 -> LogitBoost partitioned
    args(6) - File path to save the model
    */
    def main(args: Array[String]) {
        if (args.size != 7) {
            println(
                "Please provide five arguments: training data path, " +
                "test data path, sampling fraction, number of batches, " +
                "number of iterations, boolean flag, model file path."
            )
            return
        }

        val conf = new SparkConf()
        val sc = new SparkContext(conf)
        sc.setCheckpointDir("checkpoints/")

        // training
        val featureSize = Source.fromFile(args(0)).getLines().next().split(",").size - 1
        val data = sc.textFile(args(0), minPartitions=featureSize)
                     .map {line => line.split(",").map(_.trim.toDouble)}
        // TODO: does this RDD need to be repartitioned?
        val train = data.map {t => Instance((t.head + t.head - 1.0).toInt, t.tail.toVector)}
                        .cache()
        var test = sc.textFile(args(1))
                     .sample(false, 0.1)
                     .map {line => line.split(",").map(_.trim.toDouble)}
                     .map {t => Instance((t.head + t.head - 1.0).toInt, t.tail.toVector)}
                     .cache()
        println("Training data size: " + train.count)
        println("Test data size: " + test.count)
        val nodes = args(5).toInt match {
            case 1 => Controller.runADTreeWithAdaBoost(train, test, 0.05, args(2).toDouble, args(3).toInt, args(4).toInt, false)
            // TODO: added bulk learning option
            // case 2 => Controller.runADTreeWithBulkAdaboost(rdd, args(3).toInt)
            case 3 => Controller.runADTreeWithLogitBoost(train, test, 0.05, args(2).toDouble, args(3).toInt, args(4).toInt, false)
        }
        for (t <- nodes) {
            println(t)
        }

        // evaluation
        /*
        val trainMargin = train.coalesce(20).glom()
                               .map(_.map(t => SplitterNode.getScore(0, nodes.toList, t) * t.y))
                               .cache()
        val trainError = trainMargin.map(_.count(_ <= 1e-8)).reduce(_ + _)
        val trainTotal = trainMargin.map(_.size).reduce(_ + _)
        val trainErrorRate = trainError.toDouble / trainTotal
        // println("Margin: " + trainMargin.sum)
        println("Training error is " + trainErrorRate)
        */
        sc.stop()

        SplitterNode.save(nodes.toList, args(6))
    }
}
