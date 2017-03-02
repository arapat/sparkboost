package sparkboost.examples

import scala.io.Source
import scala.annotation.tailrec
import util.Random.nextDouble

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.SparseVector

import sparkboost._

object SpliceSite {
    /*
    Commandline options:

    --train       - file path to the training data
    --test        - file path to the test data
    --sample-frac - fraction for sampling
    --iteration   - number of iterations
    --depth       - max depth of the tree
    --algorithm   - algorithm selection
                        1 -> AdaBoost partitioned
                        2 -> AdaBoost non-partitioned
                        3 -> LogitBoost partitioned
    --save-model  - File path to save the model
    --load-model  - File path to load the model
    --format      - data format
                        1 -> raw data
                        2 -> objects
                        3 -> sample by model
    --train-rdd   - path to save training data RDD (row-based)
    --test-rdd    - path to save testing data RDD
    */

    // Global constants (TODO: parameterize them)
    val BINSIZE = 1
    val ALLSAMPLE = 0.05
    val NEGSAMPLE = 0.01
    // training/testing split
    val TRAIN_PORTION = 0.75
    val TEST_PORTION = 1.0 - TRAIN_PORTION

    // Construction features: P1 and P2 (as in Degroeve's SpliceMachine paper)
    val FEATURE_TYPE = 2
    val CENTER = 60
    val LEFT_WINDOW = 50  // out of 59
    val RIGHT_WINDOW = 50  // out of 80
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
               .map {case (key, value): (key.slice(2, key.size), value)}
               .toMap
    }

    def preprocessAssign(featureSize: Int)(partIndex: Int, data: Iterator[Instance]) = {
        val partId = partIndex % BINSIZE
        val sample = data.toList
        (partId until featureSize by BINSIZE).map(
            idx => (idx, (idx, sample))
        ).iterator
    }

    def preprocessMergeSort(a: (Int, List[Instance]), b: (Int, List[Instance])) = {
        val index = a._1

        @tailrec
        def mergeSort(res: List[Instance], xs: List[Instance], ys: List[Instance]): List[Instance] = {
            (xs, ys) match {
                case (Nil, _) => res.reverse ::: ys
                case (_, Nil) => res.reverse ::: xs
                case (x :: xs1, y :: ys1) => {
                    if (x.X(index) < y.X(index)) mergeSort(x :: res, xs1, ys)
                    else                         mergeSort(y :: res, xs, ys1)
                }
            }
        }

        (index, mergeSort(Nil, a._2, b._2))
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
        }.map(t => indexMap(t._1 + t._2.toString)).toArray.sorted
        Instance(data(0).toInt,
                 new SparseVector(featureSize, nonzeros, nonzeros.map(_ => 1.0)))
    }

    def main(args: Array[String]) {
        // Parse and read options
        val options = parseOptions(args)
        val trainPath = options("train")
        val testPath = options("test")
        val sampleFrac = options("sample-frac")
        val T = options("iteration")
        val depth = options("depth")
        val algo = options("algorithm")
        val modelReadPath = options.get("save-model", "")
        val modelWritePath = options.get("load-model", "")
        val dataFormat = options("format")
        val trainFile = options("train-rdd")
        val testFile = options("test-rdd")

        val conf = new SparkConf()
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.kryoserializer.buffer.mb","24")
        val sc = new SparkContext(conf)
        sc.setCheckpointDir("checkpoints/")

        val glomTrain = (
            if (loadMode == 2){
                sc.objectFile[(List[Instance], Int, List[Double])](trainObjFile)
            } else {
                val balancedTrain = sc.textFile(args(0), 1000)
                                      .map(rowToInstance)
                                      .filter(inst => {
                                          (inst.y > 0 || nextDouble() <= NEGSAMPLE)
                                      })
                val sampledTrain: RDD[Instance] = {
                    val fraction = ALLSAMPLE * TRAIN_PORTION
                    if (loadMode == 1) {
                        balancedTrain.filter(_ => nextDouble() <= fraction)
                    } else {
                        val wfunc = UpdateFunc.adaboostUpdateFunc _
                        val baseNodes = SplitterNode.load(args(7))
                        balancedTrain.mapPartitions {
                            (iterator: Iterator[Instance]) => {
                                val array = iterator.toList
                                var sampleList = Array[Instance]()
                                val size = (array.size * fraction).ceil.toInt
                                val weights = array.map(t =>
                                    wfunc(t.y, 1.0, SplitterNode.getScore(0, baseNodes, t))
                                )
                                val weightSum = weights.reduce(_ + _)
                                val segsize = weightSum.toDouble / size

                                var curWeight = nextDouble() * segsize // first sample point
                                var accumWeight = 0.0
                                for (iw <- array.zip(weights)) {
                                    while (accumWeight <= curWeight && curWeight < accumWeight + iw._2) {
                                        sampleList :+= iw._1
                                        curWeight += segsize
                                    }
                                    accumWeight += iw._2
                                }
                                sampleList.map(t => Instance.clone(t, 1.0, baseNodes)).toIterator
                            }
                        }
                    }
                }
                sampledTrain.mapPartitionsWithIndex(preprocessAssign(featureSize))
                            .map(t => (t._1, (t._2._1, t._2._2.sortWith(_.X(t._1) < _.X(t._1)))))
                            .reduceByKey(preprocessMergeSort)
                            .map(t => preprocessSlices(0.05)(t._2))
            }
        ).persist(org.apache.spark.storage.StorageLevel.MEMORY_ONLY)  // _SER)
        val glomTest = (
            if (loadMode == 1 || loadMode == 3) {
                sc.textFile(args(0), 20)
                  .map(rowToInstance)
                  .filter(inst => {
                      nextDouble() <= ALLSAMPLE * TEST_PORTION &&
                      (inst.y > 0 || nextDouble() <= NEGSAMPLE)
                  }).coalesce(20).glom()
            } else {
                sc.objectFile[Array[Instance]](testObjFile)
            }
        ).persist(org.apache.spark.storage.StorageLevel.MEMORY_ONLY)  // _SER)

        if (loadMode == 1 || loadMode == 3) {
            glomTrain.saveAsObjectFile(trainObjFile)
            glomTest.saveAsObjectFile(testObjFile)
        }

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

        val baseNodes = {
            if (loadMode == 3) SplitterNode.load(args(7))
            else               Nil
        }
        val nodes = args(5).toInt match {
            case 1 =>
                Controller.runADTreeWithAdaBoost(
                    glomTrain, glomTest, 0.05, args(2).toDouble, args(3).toInt, args(4).toInt, baseNodes
                )
            /*
            case 3 =>
                Controller.runADTreeWithLogitBoost(
                    glomTrain, glomTest, 0.05, args(2).toDouble, args(3).toInt, args(4).toInt, baseNodes
                )
            */
        }

        nodes.foreach(println)
        SplitterNode.save(nodes, modelWritePath)
        sc.stop()
    }


            // print visualization meta data for JBoost
            val posTrain = data.flatMap(_._1).filter(_.y > 0).takeSample(true, 3000)
            val negTrain = data.flatMap(_._1).filter(_.y < 0).takeSample(true, 3000)
            val posTest = glomTest.flatMap(t => t).filter(_.y > 0).takeSample(true, 3000)
            val negTest = glomTest.flatMap(t => t).filter(_.y < 0).takeSample(true, 3000)
            val esize = 6000

            val trainFile = new File("trial0.train.boosting.info")
            val trainWrite = new BufferedWriter(new FileWriter(trainFile))
            val testFile = new File("trial0.test.boosting.info")
            val testWrite = new BufferedWriter(new FileWriter(testFile))

            for (i <- 1 to nodes.size) {
                trainWrite.write(s"iteration=$i : elements=$esize : boosting_params=None (jboost.booster.AdaBoost):\n")
                testWrite.write(s"iteration=$i : elements=$esize : boosting_params=None (jboost.booster.AdaBoost):\n")
                var id = 0
                for (t <- posTrain) {
                    val score = SplitterNode.getScore(0, nodes, t, i)
                    trainWrite.write(s"$id : $score : $score : 1 : \n")
                    id = id + 1
                }
                for (t <- negTrain) {
                    val score = SplitterNode.getScore(0, nodes, t, i)
                    val negscore = -score
                    trainWrite.write(s"$id : $negscore : $score : -1 : \n")
                    id = id + 1
                }
                id = 0
                for (t <- posTest) {
                    val score = SplitterNode.getScore(0, nodes, t, i)
                    testWrite.write(s"$id : $score : $score : 1 : \n")
                    id = id + 1
                }
                for (t <- negTest) {
                    val score = SplitterNode.getScore(0, nodes, t, i)
                    val negscore = -score
                    testWrite.write(s"$id : $negscore : $score : -1 : \n")
                    id = id + 1
                }
            }

            trainWrite.close()
            testWrite.close()
}

// command:
//
// spark/bin/spark-submit --class sparkboost.examples.SpliceSite
// --master spark://ec2-54-152-1-69.compute-1.amazonaws.com:7077
// --conf spark.executor.extraJavaOptions=-XX:+UseG1GC  ./sparkboost_2.11-0.1.jar
// /train-txt /test-txt 0.05 200 1 1 ./model.bin ./base-model.bin 3 > result.txt 2> log.txt
