package sparkboost.examples

import scala.io.Source
import scala.annotation.tailrec
import util.Random.{nextDouble => rand}

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.storage.StorageLevel

import sparkboost._

object SpliceSite {
    /*
    Commandline options:

    --train         - file path to the training data
    --sample-frac   - fraction for sampling
    --iteration     - number of iterations
    --depth         - max depth of the tree
    --algorithm     - algorithm selection
                          1 -> AdaBoost partitioned
                          2 -> AdaBoost non-partitioned
                          3 -> LogitBoost partitioned
    --save-model    - File path to save the model
    --load-model    - File path to load the model
    --format        - data format
                          1 -> raw data
                          2 -> objects
                          3 -> sample by model
    --train-rdd     - path to save training data RDD (row-based)
    --test-rdd      - path to save testing data RDD
    --test-ref-rdd  - path to save referenced testing data RDD
    */

    // Global constants (TODO: parameterize them)
    val BINSIZE = 1
    val ALLSAMPLE = 0.05
    val NEGSAMPLE = 0.01
    // training/testing split
    val TRAIN_PORTION = 0.75

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
               .map {case (key, value) => (key.slice(2, key.size), value)}
               .toMap
    }

    def loadTrainData(sc: SparkContext,
                      loadMode: Int, trainPath: String, trainObjFile: String, testObjFile: String,
                      nodes: Array[SplitterNode]):
                (Array[Int], RDD[Instances], RDD[(Int, SparseVector)], RDD[(Int, SparseVector)]) = {
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
            (data(0).toInt,
             new SparseVector(featureSize, nonzeros, nonzeros.map(_ => 1.0)))
        }

        val splits =
            if (loadMode == 2){
                Array(sc.objectFile[(Int, SparseVector)](trainObjFile),
                      sc.objectFile[(Int, SparseVector)](testObjFile))
            } else {
                val balancedData = sc.textFile(trainPath)
                                     .map(rowToInstance)
                                     .filter(inst => {
                                         (inst._1 > 0 || rand() <= NEGSAMPLE)
                                     })
                val sampledData: RDD[(Int, SparseVector)] = {
                    val fraction = ALLSAMPLE
                    if (loadMode == 1) {
                        balancedData.filter(_ => rand() <= fraction)
                    } else {
                        // TODO: parameterize wfunc
                        val wfunc = UpdateFunc.adaboostUpdateFunc _
                        balancedData.mapPartitions {
                            (iterator: Iterator[(Int, SparseVector)]) => {
                                val array = iterator.toList
                                var sampleList = Array[(Int, SparseVector)]()
                                val size = (array.size * fraction).ceil.toInt
                                val weights = array.map(t =>
                                    wfunc(t._1, 1.0, SplitterNode.getScore(0, nodes, t._2))
                                )
                                val weightSum = weights.reduce {(a: Double, b: Double) => a + b}
                                val segsize = weightSum.toDouble / size

                                var curWeight = rand() * segsize // first sample point
                                var accumWeight = 0.0
                                for (iw <- array.zip(weights)) {
                                    while (accumWeight <= curWeight && curWeight < accumWeight + iw._2) {
                                        sampleList :+= iw._1
                                        curWeight += segsize
                                    }
                                    accumWeight += iw._2
                                }
                                sampleList.toIterator
                            }
                        }
                    }
                }
                sampledData.randomSplit(Array(TRAIN_PORTION, 1.0 - TRAIN_PORTION))
            }
        val (trainRaw, test): (RDD[(Int, SparseVector)], RDD[(Int, SparseVector)]) = (splits(0), splits(1))


        // apply(x: SparseVector, ptr: Array[Int], index: Int, sliceFrac: Double)
        // TODO: support SIZE > 1
        // TODO: parameterize sliceFrac
        val sliceFrac = 0.05
        val numPartitions = 160
        val y = trainRaw.map(_._1).collect()
        val train = trainRaw.zipWithIndex()
                            .flatMap {case ((y, x), idx) =>
                                        (0 until x.size).map(k => (k, (x(k), idx.toInt)))}
                            .groupByKey(numPartitions)
                            .map {case (index, xAndPtr) => {
                                val (x, ptr) = xAndPtr.toArray.sorted.unzip
                                Instances((new DenseVector(x)).toSparse, ptr, index, sliceFrac, true)
                            }}
        (y, train, trainRaw, test)
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
        val trainPath = options("train")
        // val testPath = options.get("test")
        val sampleFrac = options("sample-frac").toDouble
        val T = options("iteration").toInt
        val depth = options("depth").toInt
        val algo = options("algorithm").toInt
        val modelReadPath = options.getOrElse("save-model", "")
        val modelWritePath = options.getOrElse("load-model", "")
        val loadMode = options("format").toInt
        val trainObjFile = options.getOrElse("train-rdd", "")
        val testObjFile = options.getOrElse("test-rdd", "")
        val testRefObjFile = options.getOrElse("test-ref-rdd", "")

        val baseNodes = {
            if (loadMode == 3) SplitterNode.load(modelReadPath)
            else               Array[SplitterNode]()
        }

        val (yLocal, train, trainRaw, test) = loadTrainData(
            sc, loadMode, trainPath, trainObjFile, testObjFile, baseNodes)
        trainRaw.persist(StorageLevel.MEMORY_ONLY)
        train.persist(StorageLevel.MEMORY_ONLY)
        test.persist(StorageLevel.MEMORY_ONLY)
        val y = sc.broadcast(yLocal)
        val testRef = (
            if (testRefObjFile == "") {
                test
            } else {
                sc.objectFile[(Int, SparseVector)](testRefObjFile)
            }
        ).persist(StorageLevel.MEMORY_ONLY)

        if (loadMode == 1 || loadMode == 3) {
            if (trainObjFile != "") {
                trainRaw.saveAsObjectFile(trainObjFile)
            }
            if (testObjFile != "") {
                test.saveAsObjectFile(testObjFile)
            }
        }

        println("Train data size: " + trainRaw.count)
        println("Test data size: " + test.count)
        println("Test ref data size: " + testRef.count)
        println("Distinct positive samples in the training data: " +
                trainRaw.filter(_._1 > 0).count)
        println("Distinct negative samples in the training data: " +
                trainRaw.filter(_._1 < 0).count)
        println("Distinct positive samples in the test data: " +
                test.filter(_._1 > 0).count)
        println("Distinct negative samples in the test data: " +
                test.filter(_._1 < 0).count)
        println()

        val nodes = algo.toInt match {
            case 1 =>
                Controller.runADTreeWithAdaBoost(
                    sc, train, y, trainRaw, test, sampleFrac, T, depth, baseNodes
                )
            /*
            case 3 =>
                Controller.runADTreeWithLogitBoost(
                    sc, train, y, trainRaw, test, sampleFrac, T, depth, baseNodes
                )
            */
        }

        nodes.foreach(println)
        SplitterNode.save(nodes, modelWritePath)
        sc.stop()
    }

    /*
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
    */
}

// command:
//
// spark/bin/spark-submit --class sparkboost.examples.SpliceSite
// --master spark://ec2-54-152-1-69.compute-1.amazonaws.com:7077
// --conf spark.executor.extraJavaOptions=-XX:+UseG1GC  ./sparkboost_2.11-0.1.jar
// /train-txt /test-txt 0.05 200 1 1 ./model.bin ./base-model.bin 3 > result.txt 2> log.txt
