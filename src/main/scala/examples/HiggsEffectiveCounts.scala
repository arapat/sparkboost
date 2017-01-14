package sparkboost.examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import sparkboost._

object HiggsEffectiveCounts {
    /*
    args(0) - master node URL
    args(1) - file path to the training data
    args(2) - File path to save the model
    */
    def main(args: Array[String]) {
        if (args.size != 3) {
            println(
                "Please provide three arguments: master url, training data path, " +
                "model file path."
            )
            return
        }

        val conf = new SparkConf().setMaster(args(0))
        val sc = new SparkContext(conf)
        var rdd = sc.textFile(args(1))
                    .map {line => line.split(",").map(_.trim.toDouble)}
                    .map {t => Instance((t.head + t.head - 1.0).toInt, t.tail.toVector)}
                    .cache()

        val nodes = SplitterNode.load(args(2))
        var ec = List(rdd.count.toDouble)
        for (node <- nodes) {
            rdd = UpdateFunc.adaboostUpdate(rdd, node).cache()
            val weights = rdd.map(_.w)
            val wsum: Double = weights.reduce(_ + _)
            val wsq: Double = weights.map {x => x * x}
                                     .reduce(_ + _)
            ec = ec :+ ((wsum * wsum).toDouble / wsq)
        }
        ec.foreach(t => print(t + ", "))
    }
}
