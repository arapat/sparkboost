package sparkboost.examples

import scala.io.Source
import util.Random.nextDouble

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
    args(3) - number of iterations
    args(4) - max depth of the tree
    args(5) - algorithm selection
        1 -> AdaBoost partitioned
        2 -> AdaBoost non-partitioned
        3 -> LogitBoost partitioned
    args(6) - File path to save the model
    args(7) - data format
        1 -> raw data
        2 -> objects
    */
    def main(args: Array[String]) {
        val BINSIZE = 1
        val ALLSAMPLE = 0.05
        val NEGSAMPLE = 0.01
        val trainObjFile = "/train-pickle-onebit2/"
        val testObjFile = "/test-pickle-onebit2/"

        // Feature: P1 + P2
        val FEATURE_TYPE = 2
        val CENTER = 60
        val LEFT_WINDOW = 20 // 59
        val RIGHT_WINDOW = 20 // 80
        val WINDOW_SIZE = LEFT_WINDOW + RIGHT_WINDOW
        val featureSize =
            if (FEATURE_TYPE == 1) {
                (WINDOW_SIZE) * 4
            } else {
                (WINDOW_SIZE) * 4 + (WINDOW_SIZE - 1) * 4 * 4
            }
        val indexMap = {
            val unit = List("A", "C", "G", "T")
            val unit2 = unit.map(t => unit.map(t + _)).reduce(_ ++ _)
            val p1 = (0 until WINDOW_SIZE).map(idx => unit.map(idx + _)).reduce(_ ++ _)
            val p2 = (0 until (WINDOW_SIZE - 1)).map(idx => unit2.map(idx + _)).reduce(_ ++ _)

            if (FEATURE_TYPE == 1) p1.zip(0 until p1.size).toMap
            else (p1 ++ p2).zip(0 until (p1.size + p2.size)).toMap
        }

        def preprocessAssign(featureSize: Int)(partIndex: Int, data: Iterator[Instance]) = {
            val partId = partIndex % BINSIZE
            val sample = data.toList
            (partId until featureSize by BINSIZE).map(
                idx => (idx, (idx, sample))
            ).iterator
        }

        def preprocessMergeSort(a: (Int, List[Instance]), b: (Int, List[Instance])) = {
            if (a._2.size == 0) {
                b
            } else if (b._2.size == 0) {
                a
            } else {
                var merged = Array[Instance]()
                val index = a._1
                val leftIter = a._2.iterator
                val rightIter = b._2.iterator
                var leftItem = leftIter.next
                var rightItem = rightIter.next
                var lastLeft =
                    if (leftItem.X(index) < rightItem.X(index)) {
                        merged :+= leftItem
                        true
                    } else {
                        merged :+= rightItem
                        false
                    }
                while ((!lastLeft || leftIter.hasNext) && (lastLeft || rightIter.hasNext)) {
                    if (lastLeft) {
                        leftItem = leftIter.next
                    } else {
                        rightItem = rightIter.next
                    }
                    lastLeft =
                        if (leftItem.X(index) < rightItem.X(index)) {
                            merged :+= leftItem
                            true
                        } else {
                            merged :+= rightItem
                            false
                        }
                }
                if (lastLeft) {
                    merged :+= rightItem
                } else {
                    merged :+= leftItem
                }
                while (leftIter.hasNext) {
                    merged :+= leftIter.next
                }
                while (rightIter.hasNext) {
                    merged :+= rightIter.next
                }

                (index, merged.toList)
            }
        }

        def preprocessSlices(sliceFrac: Double)(indexData: (Int, List[Instance])) = {
            val index = indexData._1
            val data = indexData._2.map(_.X(index)).toVector
            var last = data(0)
            for (t <- data) {
                require(last <= t)
                last = t
            }
            val sliceSize = math.max(1, (indexData._2.size * sliceFrac).floor.toInt)
            val slices =
                (sliceSize until data.size by sliceSize).map(
                    idx => 0.5 * (data(idx - 1) + data(idx))
                ).distinct.toList :+ Double.MaxValue
            (indexData._2, index, slices)
        }

        def rowToInstance(s: String) = {
            val data = s.slice(1, s.size - 1).split(",")
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
            }.map(t => indexMap(t._1 + t._2.toString)).sorted
            var feature = Array[Double]()
            var last = 0
            for (i <- 0 until featureSize) {
                if (last < nonzeros.size && nonzeros(last) == i) {
                    feature :+= 1.0
                    last += 1
                } else {
                    feature :+= 0.0
                }
            }
            Instance(data(0).toInt, feature) // .toVector)
        }

        if (args.size != 8) {
            println(
                "Please provide five arguments: training data path, " +
                "test data path, sampling fraction, " +
                "number of iterations, max depth, boolean flag, model file path, " +
                "data source type."
            )
            return
        }

        val conf = new SparkConf()
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.kryoserializer.buffer.mb","24")
        val sc = new SparkContext(conf)
        val sqlContext = new SQLContext(sc)
        sc.setCheckpointDir("checkpoints/")

        // training
        val allData = sc.textFile(args(0), 200)
                        .map(rowToInstance)
                        .filter(inst => {
                            nextDouble() <= ALLSAMPLE &&
                            (inst.y > 0 || nextDouble() <= NEGSAMPLE)
                        }).persist(org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK_SER)
        allData.count()
        var Array(train, test) = allData.randomSplit(Array(0.75, 0.25))
        train = train.cache()
        test = test.cache()
        train.count()
        test.count()

        val glomTrain = (
            if (args(7).toInt == 1) {
                // up-sample positive samples
                /*
                val perPart = math.min(featureSize, 200.0) // TODO: make this a variable
                val posSize = 3000
                val dupSize = 1  // (perPart * featureSize / posSize).ceil.toInt

                train.flatMap(inst =>
                    if (inst.y < 0) {
                        Iterator(inst)
                    } else {
                        (0 until dupSize).map(_ => Instance(inst.y, inst.X))
                    }
                )
                */
                train.mapPartitionsWithIndex(preprocessAssign(featureSize))
                     .map(t => (t._1, (t._2._1, t._2._2.sortWith(_.X(t._1) < _.X(t._1)))))
                     .reduceByKey(preprocessMergeSort)
                     .map(t => preprocessSlices(0.05)(t._2))
            } else {
                sc.objectFile[(List[Instance], Int, List[Double])](trainObjFile)
            }
        ).persist(org.apache.spark.storage.StorageLevel.MEMORY_ONLY)  // _SER)
        val glomTest = (
            if (args(7).toInt == 1) {
                // sc.textFile(args(1))
                  // .sample(false, 0.1)
                test.coalesce(20)
                    .glom()
            } else {
                sc.objectFile[Array[Instance]](testObjFile)
            }
        ).persist(org.apache.spark.storage.StorageLevel.MEMORY_ONLY)  // _SER)
        if (args(7).toInt == 1) {
            glomTrain.saveAsObjectFile(trainObjFile)
            glomTest.saveAsObjectFile(testObjFile)
        }

        // println("Training data size: " + train.count)

        println("Train partition (set) size: " + glomTrain.count)
        println("Test partition (set) size: " + glomTest.count)
        println("Distinct positive samples in the training data: " +
                glomTrain.filter(_._2 < BINSIZE).map(_._1.count(t => t.y > 0)).reduce(_ + _))
        println("Distinct negative samples in the training data: " +
                glomTrain.filter(_._2 < BINSIZE).map(_._1.count(t => t.y < 0)).reduce(_ + _))
        println("Distinct positive samples in the test data: " +
                glomTest.map(_.count(t => t.y > 0)).reduce(_ + _))
        println("Distinct negative samples in the test data: " +
                glomTest.map(_.count(t => t.y < 0)).reduce(_ + _))
        println()
        allData.unpersist()
        train.unpersist()
        test.unpersist()

        /*
        val reducedGlomTrain = glomTrain.map(t => {
            val sample = t._1.filter(inst => {
                if (inst.y > 0) true
                else if (nextDouble() <= 0.1) true
                else false
            })
            (sample, t._2, t._3)
        }).cache()
        println("Sample size: " + reducedGlomTrain.map(_._1.size).reduce(_ + _))
        */

        // println("Test data size: " + test.count)
        // println("Positive: " + test.filter(_.y > 0).count)
        // println("Negative: " + test.filter(_.y < 0).count)

        val nodes = args(5).toInt match {
            case 1 => Controller.runADTreeWithAdaBoost(glomTrain, glomTest, 0.05, args(2).toDouble, args(3).toInt, args(4).toInt)
            case 3 => Controller.runADTreeWithLogitBoost(glomTrain, glomTest, 0.05, args(2).toDouble, args(3).toInt, args(4).toInt)
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

        SplitterNode.save(nodes, args(6))
    }
}

// command:
//
// spark/bin/spark-submit --class sparkboost.examples.SpliceSite
// --master spark://ec2-54-152-1-69.compute-1.amazonaws.com:7077
// --conf spark.executor.extraJavaOptions=-XX:+UseG1GC  ./sparkboost_2.11-0.1.jar
// /train-1m /test-txt 0.05 1 50 1 ./model.bin 2 > result.txt 2> log.txt
