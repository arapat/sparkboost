package sparkboost.examples

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import sparkboost._

object SpliceSiteAnalysis {
    /*
    Commandline options:

    --test          - file path to the test data
    --load-model    - File path to load the model
    */
    type TestRDDType = RDD[(Int, SparseVector)]

    def parseOptions(options: Array[String]) = {
        options.zip(options.slice(1, options.size))
               .zip(0 until options.size).filter(_._2 % 2 == 0).map(_._1)
               .map {case (key, value) => (key.slice(2, key.size), value)}
               .toMap
    }

    def printStats(test: TestRDDType, nodes: Array[SplitterNode]) {
        // manual fix the auPRC computation bug in MLlib
        def adjust(points: Array[(Double, Double)]) = {
            require(points.length == 2)
            require(points.head == (0.0, 1.0))
            val y = points.last
            y._1 * (y._2 - 1.0) / 2.0
        }

        // Part 1 - Compute auPRC
        val predictionAndLabels = test.map {case t =>
            (SplitterNode.getScore(0, nodes, t._2).toDouble, t._1.toDouble)
        }.cache()

        val metrics = new BinaryClassificationMetrics(predictionAndLabels)
        val auPRC = metrics.areaUnderPR + adjust(metrics.pr.take(2))

        println("Testing auPRC = " + auPRC)
        println("Testing PR = " + metrics.pr.collect.toList)
    }

    def main(args: Array[String]) {
        // Define SparkContext
        val conf = new SparkConf()
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.kryoserializer.buffer.mb","24")
        val sc = new SparkContext(conf)

        // Parse and read options
        val options = parseOptions(args)
        val testPath = options("test")
        val modelReadPath = options("load-model")

        val nodes = SplitterNode.load(modelReadPath)
        val data = sc.textFile(testPath).map(InstanceFactory.rowToInstance).cache()
        println("Distinct positive samples in the training data (test data): " +
            data.filter(_._1 > 0).count)
        println("Distinct negative samples in the training data (test data): " +
            data.filter(_._1 < 0).count)
        printStats(data, nodes)

        sc.stop()
    }
}

// command:
//
// ./spark/bin/spark-submit --master spark://ec2-54-152-198-27.compute-1.amazonaws.com:7077
// --class sparkboost.examples.SpliceSite --conf spark.executor.extraJavaOptions=-XX:+UseG1GC
// ./sparkboost_2.11-0.1.jar --test /test-txt --load-model ./model.bin
