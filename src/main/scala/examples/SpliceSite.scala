package sparkboost.examples

import collection.mutable.ArrayBuffer
import scala.io.Source
import scala.annotation.tailrec
import util.Random.{nextDouble => rand}

import org.apache.spark.Partitioner
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.storage.StorageLevel

import sparkboost._
import sparkboost.utils.Comparison


/*
Download data from S3:

    s3helper.open_bucket("ucsd-data")
    s3helper.s3_to_hdfs("splice-sites/test-txt/", "/test-txt/")
    s3helper.s3_to_hdfs("splice-sites/train-50m-txt/", "/train-txt")

*/

class UniformPartitioner(val numOfPartitions: Int, val featureSize: Int) extends Partitioner {
    def numPartitions() = numOfPartitions

    def getPartition(key: Any): Int = {
        val (batchId, index) = key.asInstanceOf[(Long, Int)]
        return (batchId.toInt * featureSize + index) % numOfPartitions
    }
}

object InstanceFactory {
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
}

object SpliceSite extends Comparison {
    /*
    Commandline options:

    Required:
        --train                - file path to the training data
        --test                 - file path to the test data
        --sample-frac          - fraction for sampling for a single batch
        --num-slices           - number of slices a feature dimension to be divded into
        --max-iteration        - maximum number of iterations (0 for unlimited)
        --algorithm            - algorithm selection
                                     1 -> AdaBoost
                                     2 -> LogitBoost (not supported yet)
        --save-model           - file path to save the model
        --save-train-rdd       - file path to save training data RDD (row-based)
        --save-train-csc-rdd   - file path to save training csc RDD (col-based)
        --save-test-rdd        - file path to save test data RDD
        --data-source          - source of data
                                     1 -> read from text files and get a new sample
                                     2 -> read from object files of an existing sample (requires additional parameters, see below)
        --improve              - percentage of improvement expected for re-sampling range in (0.0, 1.0)

    Optional:
        --improve-window       - number of iterations to wait before declaring overfitting/underfitting
        --cores                - number of cores available for this job
        --load-model           - file path to load the model
        --model-length         - length of the loaded model (for handling rollback)
        --last-resample        - the last tree node before resampling (1-based)
        --last-depth           - the depth of the tree before existing

    If "--data-source" is set to 2, the following optional parameters are required.

        --load-train-rdd       - path to read training data RDD (row-based)
        --load-train-csc-rdd   - path to read training csc RDD (col-based)
        --load-test-rdd        - path to save testing data RDD
    */

    // TODO: support BINSIZE > 1
    val BINSIZE = 1
    // training/testing split
    val TRAIN_PORTION = 0.75

    type BaseInstance = (Int, SparseVector)
    type BaseToCSCFunc = RDD[BaseInstance] => RDD[Instances]

    def parseOptions(options: Array[String]) = {
        options.zip(options.slice(1, options.size))
               .zip(0 until options.size).filter(_._2 % 2 == 0).map(_._1)
               .map {case (key, value) => (key.slice(2, key.size), value)}
               .toMap
    }

    def sampleData(trainInstance: RDD[BaseInstance], sampleFrac: Double, getWeight: (Int, Double, Double) => Double,
                   url: String, trainSavePath: String, trainCSCSavePath: String, testSavePath: String,
                   baseToCSCFunc: BaseToCSCFunc)(nodes: Array[SplitterNode]) = {
        val weightsTrain = trainInstance.map(t =>
                            (getWeight(t._1, 1.0, SplitterNode.getScore(0, nodes, t._2)), t)
                           ).cache

        val sumWeight = weightsTrain.map(_._1).reduce(_ + _)
        val posWeight = weightsTrain.filter(_._2._1 > 0).map(_._1).reduce(_ + _)
        val negWeight = sumWeight - posWeight
        val posScale = negWeight / sumWeight
        val negScale = posWeight / sumWeight
        val scaledSumWeight = 2 * posWeight * negWeight / sumWeight

        val sampleSize = (weightsTrain.count * sampleFrac).ceil.toInt
        val segSize = scaledSumWeight / sampleSize
        val offset = rand() * segSize

        val partitionSum = weightsTrain.mapPartitions((iterator: Iterator[(Double, BaseInstance)]) => {
            List(
                iterator.map { case (w, t) => (if (t._1 > 0) posScale else negScale) * w }
                        .fold(0.0)(_ + _)
            ).toIterator
        }).collect().toArray.scanLeft(0.0)(_ + _)
        val weightedSample = weightsTrain.mapPartitionsWithIndex(
            (index: Int, iterator: Iterator[(Double, BaseInstance)]) => {
                val array = iterator.toList
                var sampleList = List[BaseInstance]()
                if (array.size > 0) {
                    var accumWeight = partitionSum(index) - offset
                    var accumIndex = (accumWeight / segSize).floor.toInt
                    for (iw <- array) {
                        accumWeight += (if (iw._2._1 > 0) posScale else negScale) * iw._1
                        val end = (accumWeight / segSize).floor.toInt
                        if (end >= 0) {
                            for (_ <- 0 until (end - accumIndex)) {
                                sampleList = iw._2 +: sampleList
                            }
                        }
                        accumIndex = end
                    }
                }
                sampleList.toIterator
            }
        ).cache()

        val splits = weightedSample.randomSplit(Array(TRAIN_PORTION, 1.0 - TRAIN_PORTION))
        val (train, test): (RDD[BaseInstance], RDD[BaseInstance]) = (splits(0), splits(1))
        train.setName("sampled train data")
        test.setName("sampled test data")
        train.cache()
        test.cache()
        train.count()
        val trainCSC = baseToCSCFunc(train)

        val hadoopConf = new org.apache.hadoop.conf.Configuration()
        val hdfs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI(s"hdfs://$url"), hadoopConf)
        try {
            hdfs.delete(new org.apache.hadoop.fs.Path(trainSavePath), true)
            hdfs.delete(new org.apache.hadoop.fs.Path(testSavePath), true)
            hdfs.delete(new org.apache.hadoop.fs.Path(trainCSCSavePath), true)
            train.saveAsObjectFile(trainSavePath)
            test.saveAsObjectFile(testSavePath)
            trainCSC.saveAsObjectFile(trainCSCSavePath)
            println("Re-sampled data saved.")
        } catch {
            case e : Throwable => {
                println("Failed to save the resampled data.")
                println("Error: " + e)
            }
        }

        weightsTrain.unpersist()
        (train, test, trainCSC)
    }

    // Row store to Column Store Compression
    def baseToCSC(numSlices: Int, numCores: Int)(train: RDD[BaseInstance]) = {
        type T = (Int, Double)
        @scala.annotation.tailrec
        def merge(ls: List[T], rs: List[T], acc: List[T] = List()): List[T] = (ls, rs) match {
            case (Nil, _) => acc.reverse ++ rs
            case (_, Nil) => acc.reverse ++ ls
            case (l :: ls1, r :: rs1) =>
                if (l._1 < r._1) merge(ls1, rs, l +: acc)
                else merge(ls, rs1, r +: acc)
        }

        val trainSize = train.count.toInt
        val dim = train.first._2.size
        println(s"Feature size = $dim")
        // TODO:
        //      1. Only support 1 batch now
        //          ==>  ((idx * BINSIZE / trainSize, k), (idx, x(k))))}
        //      2. May need to shuffle the training data
        val trainCSC = train.zipWithIndex()
                            .mapPartitions(it => {
                                val t = it.toList
                                (0 until dim).map(idx => {
                                    (idx, t.map(d => (d._2.toInt, d._1._2(idx)))
                                           .filter(e => compare(e._2) != 0)
                                           .toList
                                           .sorted)
                                }).toIterator
                            }).reduceByKey((a, b) => merge(a, b))
                            .map {case (index, indVal) => {
                                val (indices, values) = indVal.unzip  // no need to sort
                                Instances(0, new SparseVector(trainSize, indices.toArray, values.toArray),
                                          index, numSlices, true)
                            }}.cache()
        trainCSC.setName("sampled train CSC data")
        trainCSC
    }

    def main(args: Array[String]) {
        // Define SparkContext
        val conf = new SparkConf()
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.kryoserializer.buffer.max", "1024m")
        val sc = new SparkContext(conf)
        sc.setCheckpointDir("/checkpoint/")

        // Parse and read options
        val options = parseOptions(args)
        val trainPath = options("train")
        val testPath = options("test")
        val sampleFrac = options("sample-frac").toDouble
        val numSlices = options("num-slices").toInt
        val maxIters = options("max-iteration").toInt
        val algo = options("algorithm").toInt
        val modelWritePath = options("save-model")
        val trainSavePath = options("save-train-rdd")
        val trainCSCSavePath = options("save-train-csc-rdd")
        val testSavePath = options("save-test-rdd")
        val source = options("data-source").toInt
        val improveFact = options("improve").toDouble
        // Optional options
        val improveWindow = options.getOrElse("improve-window", "100").toInt
        val numCores = options.getOrElse("cores", sc.defaultParallelism.toString).toInt
        val modelReadPath = options.getOrElse("load-model", "")
        val modelLength = options.getOrElse("model-length", "0").toInt
        val lastResample = options.getOrElse("resample-node", "0").toInt
        val lastDepth = options.getOrElse("last-depth", "1").toInt
        println(s"Number of cores is set to $numCores")

        val trainInstance = sc.textFile(trainPath, minPartitions=numCores)
                              .map(InstanceFactory.rowToInstance)
                              .map(t => (rand, t))
                              .sortByKey()
                              .map(t => t._2)
                              .cache
        trainInstance.setName("all train data")
        val loadNodes = {
            if (modelReadPath != "") SplitterNode.load(modelReadPath)
            else                     Array[SplitterNode]()
        }
        val baseNodes = {
            if (modelLength > 0) loadNodes.take(modelLength)
            else                 loadNodes
        }
        val hdfsURL = sc.master.split("://")(1).split(":")(0) + ":9000"
        val rowToColFunc = baseToCSC(numSlices, numCores) _
        val curSampleFunc = sampleData(trainInstance, sampleFrac, UpdateFunc.adaboostWeightUpdate,
                                       hdfsURL, trainSavePath, trainCSCSavePath, testSavePath, rowToColFunc) _
        val (train, test, trainCSC) =
            if (source == 2) {
                val trainObjFile = options("load-train-rdd")
                val testObjFile = options("load-test-rdd")
                val trainCSCObjectFile = options("load-train-csc-rdd")
                (
                    sc.objectFile[BaseInstance](trainObjFile).cache(),
                    sc.objectFile[BaseInstance](testObjFile).cache(),
                    sc.objectFile[Instances](trainCSCObjectFile).cache()
                )
            } else {
                curSampleFunc(baseNodes)
            }
        train.setName("sampled train data")
        test.setName("sampled test data")
        trainCSC.setName("sampled train CSC data")
        val y = sc.broadcast(train.map(_._1).collect)
        val testRef = sc.textFile(testPath, minPartitions=numCores)
                        .map(InstanceFactory.rowToInstance)
                        .cache()
        testRef.setName("all test data")

        println("Train data size: " + train.count)
        println("Test data size: " + test.count)
        println("Referenced test data size: " + testRef.count)
        println("Distinct positive samples in the training data: " +
                train.filter(_._1 > 0).count)
        println("Distinct negative samples in the training data: " +
                train.filter(_._1 < 0).count)
        println("Distinct positive samples in the test data: " +
                test.filter(_._1 > 0).count)
        println("Distinct negative samples in the test data: " +
                test.filter(_._1 < 0).count)
        println("CSC storage length: " + trainCSC.count)
        println()

        val nodes = algo.toInt match {
            case 1 => {
                val controller = new Controller(
                    sc,
                    curSampleFunc,
                    Learner.partitionedGreedySplit,
                    UpdateFunc.adaboostUpdate,
                    LossFunc.lossfunc,
                    UpdateFunc.adaboostWeightUpdate,
                    improveFact,
                    improveWindow,
                    modelWritePath,
                    maxIters
                )
                controller.setDatasets(train, trainCSC, y, test, testRef)
                controller.setNodes(baseNodes, lastResample, lastDepth)
                controller.runADTree
            }
            // TODO: Support LogitBoost
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
// ./spark/bin/spark-submit --master spark://ec2-54-89-40-11.compute-1.amazonaws.com:7077 \
// --class sparkboost.examples.SpliceSite --conf spark.executor.extraJavaOptions=-XX:+UseG1GC \
// ./sparkboost_2.11-0.1.jar --train /train-txt --test /test-txt --sample-frac 0.1 \
// --cores 80 --num-slices 2 --max-iteration 0 --algorithm 1 --save-model ./model.bin \
// --save-train-rdd /train0 --save-test-rdd /test0 --data-source 1 \
// --improve 0.01
