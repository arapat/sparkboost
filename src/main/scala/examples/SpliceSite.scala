package sparkboost.examples

import scala.io.Source
import collection.mutable.ArrayBuffer

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.Row

import sparkboost._

object SpliceSite {
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
        // Feature: P1 + P2
        val CENTER = 60
        val LEFT_WINDOW = 6
        val RIGHT_WINDOW = 7
        val WINDOW_SIZE = LEFT_WINDOW + RIGHT_WINDOW
        val featureSize = (WINDOW_SIZE) * 4 + (WINDOW_SIZE - 1) * 4 * 4
        val indexMap = {
            val unit = List("A", "C", "G", "T")
            val unit2 = unit.map(t => unit.map(t + _)).reduce(_ ++ _)
            val p1 = (0 until WINDOW_SIZE).map(idx => unit.map(idx + _)).reduce(_ ++ _)
            val p2 = (0 until (WINDOW_SIZE - 1)).map(idx => unit2.map(idx + _)).reduce(_ ++ _)
            (p1 ++ p2).zip(0 until (p1.size + p2.size)).toMap
        }
        def rowToInstance(t: Row) = {
            val raw = t.toSeq.tail.toVector.map(_.asInstanceOf[String])
            val window = (
                raw.slice(CENTER - 1 - LEFT_WINDOW, CENTER - 1) ++
                raw.slice(CENTER + 1, CENTER + RIGHT_WINDOW + 1)
            )
            val nonzeros = (
                (0 until WINDOW_SIZE).zip(window) ++
                (0 until WINDOW_SIZE).zip(window.zip(window.tail).map(t => t._1 + t._2))
            ).map(t => indexMap(t._1 + t._2)).sorted
            val feature = ArrayBuffer[Double]()
            var last = 0
            for (i <- 0 until featureSize) {
                if (last < nonzeros.size && nonzeros(last) == i) {
                    feature.append(1.0)
                    last += 1
                } else {
                    feature.append(0.0)
                }
            }
            Instance(t.getInt(0), feature.toVector)
        }

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
        val sqlContext = new SQLContext(sc)
        sc.setCheckpointDir("checkpoints/")

        // training
        /*
        val data = sqlContext.read.parquet(args(0)).rdd.repartition(featureSize)
        val train = data.map(rowToInstance).cache()
        var test = sqlContext.read.parquet(args(1)).rdd
                             .sample(false, 0.1)
                             .map(rowToInstance)
                             .cache()
        train.saveAsObjectFile("train-pickle/")
        test.saveAsObjectFile("test-pickle/")
        */
        val train = sc.objectFile[Instance]("/user/ec2-user/train-pickle/")
        val test = sc.objectFile[Instance]("/user/ec2-user/test-pickle/")
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
        val trainError = trainMargin.map(_.filter(_ <= 1e-8).size).reduce(_ + _)
        val trainTotal = trainMargin.map(_.size).reduce(_ + _)
        val trainErrorRate = trainError.toDouble / trainTotal
        // println("Margin: " + trainMargin.sum)
        println("Training error is " + trainErrorRate)
        */
        sc.stop()

        SplitterNode.save(nodes.toList, args(6))
    }
}
