package sparkboost

import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Higgs {
    type Instance = (Int, Vector[Double], Double)

    /*
    args(0) - file path to the training data
    args(1) - number of iterations
    */
    def main(args: Array[String]) {
        val conf = new SparkConf().setMaster("local[2]")
        val sc = new SparkContext(conf)
        sc.setCheckpointDir("checkpoint/")

        // training
        val featureSize = Source.fromFile(args(0)).getLines().next().split(",").size - 1
        val data = sc.textFile(args(0), minPartitions=featureSize)
                     .map {line => line.split(",").map(_.trim.toDouble)}
        // TODO: does this RDD need to be repartitioned?
        val rdd = data.map {t => ((t.head + t.head - 1.0).toInt, t.tail.toVector, 1.0)}
                      .cache()
        println("Training data size: " + rdd.count)
        val nodes = Controller.runADTreeWithAdaBoost(rdd, args(1).toInt, false)
        for (t <- nodes) {
            println(t)
        }

        // evaluation
        val trainMargin = rdd.map {t => (SplitterNode.getScore(0, nodes.toList, t._2) * t._1)}
                             .cache()
        val trainError = (trainMargin.filter{_ <= 0}.count).toDouble / trainMargin.count()
        println("Margin " + trainMargin.take(10).toList)
        println("Training error is " + trainError)
        sc.stop()
    }
}
