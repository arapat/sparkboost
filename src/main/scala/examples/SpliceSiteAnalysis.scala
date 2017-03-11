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

    // Global constants (TODO: parameterize them)
    val BINSIZE = 1
    val ALLSAMPLE = 0.20
    val NEGSAMPLE = 0.0029
    // training/testing split
    val TRAIN_PORTION = 0.75

    // Construction features: P1 and P2 (as in Degroeve's SpliceMachine paper)
    val FEATURE_TYPE = 1
    val CENTER = 60
    val LEFT_WINDOW = 59  // out of 59
    val RIGHT_WINDOW = 80  // out of 80
    val WINDOW_SIZE = LEFT_WINDOW + RIGHT_WINDOW
    val featureSize = (WINDOW_SIZE) * 4 + {if (FEATURE_TYPE == 1) 0 else (WINDOW_SIZE - 1) * 4 * 4}
    val indexMap = {
        val unit = List("A", "C", "G", "T")
        val unit2 = unit.map(t => unit.map(t + _)).reduce(_ ++ _)
        val p1 = (0 until WINDOW_SIZE).map(idx => unit.map(idx + _)).reduce(_ ++ _)
        val p2 = (0 until (WINDOW_SIZE - 1)).map(idx => unit2.map(idx + _)).reduce(_ ++ _)

        if (FEATURE_TYPE == 1) p1.zip(0 until p1.size).toMap
        else (p1 ++ p2).zip(0 until (p1.size + p2.size)).toMap
    }

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

    def rowToInstance(s: String) = {
        val data = s.slice(1, s.size - 2).split(", u'")
        val raw = data(1)
        val window = (
            raw.slice(CENTER - 1 - LEFT_WINDOW, CENTER - 1) ++
            raw.slice(CENTER + 1, CENTER + RIGHT_WINDOW + 1)
        )
        val nonzeros = {
            if (FEATURE_TYPE == 1) (0 until WINDOW_SIZE).zip(window)
            else {
                (0 until WINDOW_SIZE).zip(window) ++
                (0 until WINDOW_SIZE).zip(
                    window.zip(window.tail).map(t => t._1.toString + t._2)
                )
            }
        }.map(t => indexMap(t._1 + t._2.toString)).toArray.sorted
        (data(0).toInt,
         new SparseVector(featureSize, nonzeros, nonzeros.map(_ => 1.0)))
    }

    def main(args: Array[String]) {
        // Define SparkContext
        val conf = new SparkConf()
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.kryoserializer.buffer.mb","24")
        val sc = new SparkContext(conf)
        // TODO: delete checkpoints before exiting
        sc.setCheckpointDir("checkpoints/")

        // Parse and read options
        val options = parseOptions(args)
        val testPath = options("test")
        val modelReadPath = options("load-model")

        val nodes = SplitterNode.load(modelReadPath)
        val data = sc.textFile(testPath).map(rowToInstance).cache()
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
