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
    */
    def main(args: Array[String]) {
        if (args.size != 4) {
            println(
                "Please provide four arguments: master url, training data path, " +
                "testing data path, model file path."
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

        val nodes = SplitterNode.load(args(3))
        var trainError = ListBuffer[Double]()
        var testError = ListBuffer[Double]()
        var ec = ListBuffer(train.count.toDouble)
        var iters = 0
        for (node <- nodes) {
            iters = iters + 1
            train = UpdateFunc.adaboostUpdate(train, node).cache()
            test = UpdateFunc.adaboostUpdate(test, node).cache()
            if (iters % 25 == 0) {
                train.checkpoint()
                test.checkpoint()
            }

            trainError += train.filter(t => compare(t.w, 1.0) >= 0).count.toDouble / trainSize
            testError += test.filter(t => compare(t.w, 1.0) >= 0).count.toDouble / testSize
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
