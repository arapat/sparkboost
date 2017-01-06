package sparkboost

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Higgs {
    type Instance = (Int, Vector[Double], Double)

    def main(args: Array[String]) {
        val conf = new SparkConf().setMaster("local[2]")
        val sc = new SparkContext(conf)
        sc.setCheckpointDir("checkpoint/")
        val data = sc.textFile(args(0))
                     .map {line => line.split(",").map(_.trim.toDouble)}
        val rdd = data.map {t => (t.head, t.tail.toVector)}
                      .map {t => ((t._1 + t._1 - 1.0).toInt, t._2, 1.0)}
                      .cache()
        Controller.runADTreeWithAdaBoost(rdd, 10, false)
        sc.stop()
    }
}
