package sparkboost.examples

import math.exp
import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.Row

import sparkboost._

object SpliceSiteAnalysis {
    type RDDType = RDD[(List[Instance], Int, List[Double])]
    type TestRDDType = RDD[Array[Instance]]

    def printEffCnt(train: RDDType, nodes: List[SplitterNode]) = {
        def addTwoArrays(a: Array[(Double, Double)], b: Array[(Double, Double)]) = {
            a.zip(b).map {case (t1, t2) => (t1._1 + t2._1, t1._2 + t2._2)}
        }

        def getWeights(inst: Instance) = {
            (1 to nodes.size).map(i => inst.y * SplitterNode.getScore(0, nodes, inst, i))
                                .map(s => exp(-s))
                                .map(w => (w, w * w))
                                .toArray
        }

        val cnt = train.map(_._1.size).reduce(_ + _)
        val wwsq = train.map(_._1)
                        .map(_.map(getWeights).reduce(addTwoArrays))
                        .reduce(addTwoArrays)
        val posCnt = train.map(_._1.count(_.y > 0)).reduce(_ + _)
        val posWWsq = train.map(_._1.filter(_.y > 0))
                           .map(_.map(getWeights).reduce(addTwoArrays))
                           .reduce(addTwoArrays)

        val negCnt = train.map(_._1.count(_.y < 0)).reduce(_ + _)
        val negWWsq = train.map(_._1.filter(_.y < 0))
                           .map(_.map(getWeights).reduce(addTwoArrays))
                           .reduce(addTwoArrays)
        wwsq.foreach {case (w, wsq) => println((w * w) / wsq / cnt)}
        println()
        posWWsq.foreach {case (w, wsq) => println((w * w) / wsq / posCnt)}
        println()
        negWWsq.foreach {case (w, wsq) => println((w * w) / wsq / negCnt)}
    }

    def main(args: Array[String]) {
        val BINSIZE = 1
        val ALLSAMPLE = 0.05
        val NEGSAMPLE = 0.01
        val trainObjFile = "/train-pickle-onebit2/"
        val testObjFile = "/test-pickle-onebit2/"

        val conf = new SparkConf()
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.kryoserializer.buffer.mb","24")
        val sc = new SparkContext(conf)
        val sqlContext = new SQLContext(sc)
        sc.setCheckpointDir("checkpoints/")

        val glomTrain = sc.objectFile[(List[Instance], Int, List[Double])](trainObjFile)
                          .persist(org.apache.spark.storage.StorageLevel.MEMORY_ONLY)  // _SER)
        val glomTest = sc.objectFile[Array[Instance]](testObjFile)
                         .persist(org.apache.spark.storage.StorageLevel.MEMORY_ONLY)
        val nodes = SplitterNode.load(args(0))
        printEffCnt(glomTrain, nodes)

        // print visualization meta data for JBoost
        val posTrain = glomTrain.flatMap(_._1).filter(_.y > 0).takeSample(true, 1000)
        val negTrain = glomTrain.flatMap(_._1).filter(_.y < 0).takeSample(true, 3000)
        val posTest = glomTest.flatMap(t => t).filter(_.y > 0).takeSample(true, 1000)
        val negTest = glomTest.flatMap(t => t).filter(_.y < 0).takeSample(true, 3000)
        val esize = 4000

        val trainFile = new File("trial0.train.boosting.info")
        val trainWrite = new BufferedWriter(new FileWriter(trainFile))
        val testFile = new File("trial0.test.boosting.info")
        val testWrite = new BufferedWriter(new FileWriter(testFile))
        val lnodes = nodes.toList

        for (i <- 1 to nodes.size) {
            trainWrite.write(s"iteration=$i : elements=$esize : boosting_params=None (jboost.booster.AdaBoost):\n")
            testWrite.write(s"iteration=$i : elements=$esize : boosting_params=None (jboost.booster.AdaBoost):\n")
            var id = 0
            for (t <- posTrain) {
                val score = SplitterNode.getScore(0, lnodes, t, i)
                trainWrite.write(s"$id : $score : $score : 1 : \n")
                id = id + 1
            }
            for (t <- negTrain) {
                val score = SplitterNode.getScore(0, lnodes, t, i)
                val negscore = -score
                trainWrite.write(s"$id : $negscore : $score : -1 : \n")
                id = id + 1
            }
            id = 0
            for (t <- posTest) {
                val score = SplitterNode.getScore(0, lnodes, t, i)
                testWrite.write(s"$id : $score : $score : 1 : \n")
                id = id + 1
            }
            for (t <- negTest) {
                val score = SplitterNode.getScore(0, lnodes, t, i)
                val negscore = -score
                testWrite.write(s"$id : $negscore : $score : -1 : \n")
                id = id + 1
            }
        }

        trainWrite.close()
        testWrite.close()

        sc.stop()
    }
}

// command:
//
// spark/bin/spark-submit --class sparkboost.examples.SpliceSite
// --master spark://ec2-54-152-1-69.compute-1.amazonaws.com:7077
// --conf spark.executor.extraJavaOptions=-XX:+UseG1GC  ./sparkboost_2.11-0.1.jar
// /train-1m /test-txt 0.05 1 50 1 ./model.bin 2 > result.txt 2> log.txt
