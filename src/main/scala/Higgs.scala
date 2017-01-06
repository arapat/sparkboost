package sparkboost

import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Higgs {
    type Instance = (Int, Vector[Double], Double)

    def main(args: Array[String]) {
        val conf = new SparkConf().setMaster("local[2]")
        val sc = new SparkContext(conf)
        sc.setCheckpointDir("checkpoint/")

        // training
        val featureSize = Source.fromFile(args(0)).getLines().next().split(",").size - 1
        val data = sc.textFile(args(0))
                     .map {line => line.split(",").map(_.trim.toDouble)}
        val rdd = data.map {t => (t.head, t.tail.toVector)}
                      .map {t => ((t._1 + t._1 - 1.0).toInt, t._2, 1.0)}
                      .repartition(featureSize)
                      .cache()
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
