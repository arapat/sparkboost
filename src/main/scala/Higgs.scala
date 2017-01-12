package sparkboost

import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Higgs {
    /*
    args(0) - master node URL
    args(1) - file path to the training data
    args(2) - number of iterations
    args(3) - Boolean flag for using the partitioned algorithm
    */
    def main(args: Array[String]) {
        val conf = new SparkConf().setMaster(args(0))
        val sc = new SparkContext(conf)
        sc.setCheckpointDir("checkpoint/")

        // training
        val featureSize = Source.fromFile(args(1)).getLines().next().split(",").size - 1
        val data = sc.textFile(args(1), minPartitions=featureSize)
                     .map {line => line.split(",").map(_.trim.toDouble)}
        // TODO: does this RDD need to be repartitioned?
        val rdd = data.map {t => Instance((t.head + t.head - 1.0).toInt, t.tail.toVector)}
                      .cache()
        println("Training data size: " + rdd.count)
        val nodes =
            if (args(3).toBoolean) {
                Controller.runADTreeWithAdaBoost(rdd, args(2).toInt, false)
            } else {
                Controller.runADTreeWithBulkAdaboost(rdd, args(2).toInt)
            }
        for (t <- nodes) {
            println(t)
        }

        // evaluation
        val trainMargin = rdd.map {t => (SplitterNode.getScore(0, nodes.toList, t.X) * t.y)}
                             .cache()
        val trainError = (trainMargin.filter{_ <= 0}.count).toDouble / trainMargin.count()
        println("Margin " + trainMargin.take(10).toList)
        println("Training error is " + trainError)
        sc.stop()
    }
}
