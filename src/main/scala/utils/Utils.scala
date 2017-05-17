package sparkboost.utils

import math.exp
import math.log
import math.max
import math.sqrt

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import sparkboost.SplitterNode
import sparkboost.Types


object Utils extends Comparison {
    // Limit the prediction score in a reasonable range
    def safeLogRatio(a: Double, b: Double) = {
        if (compare(a) == 0 && compare(b) == 0) {
            0.0
        } else {
            val ratio = math.min(10.0, math.max(a / b, 0.1))
            log(ratio)
        }
    }

    /*
    def getThreshold(gamma: Double, delta: Double)(wsum: Double) = {
        // val rho = log((0.5 + 2 * gamma) * (0.5 - gamma) / (0.5 - 2 * gamma) / (0.5 + gamma))
        // val alpha = 1.0 / rho * log((1.0 - delta) / delta)
        // val beta = 1.0 / rho * log((0.5 - gamma) / (0.5 - 2 * gamma))
        // 2 * alpha + (2 * beta  - 1) * wsum
        val kbr = 1
        val n = max(100.0, wsum)
        sqrt(kbr * n * log(log(n) / delta)) + 2 * n * gamma
    }
    */

    def printStats(train: Types.BaseRDD, glomTrain: Types.TrainRDDType,
                   test: Types.BaseRDD, testRef: Types.BaseRDD,
                   localNodes: Array[SplitterNode], iteration: Int,
                   lastResample: Int = 0) = {
        // manual fix the auPRC computation bug in MLlib
        def adjust(points: Array[(Double, Double)]) = {
            require(points.length == 2)
            require(points.head == (0.0, 1.0))
            val y = points.last
            y._1 * (y._2 - 1.0) / 2.0
        }

        // TODO: extend to other boosting loss functions
        def getLossFunc(predictionAndLabels: RDD[(Double, Double)]) = {
            val scores = predictionAndLabels.map(t => (t._2, exp(-t._1 * t._2))).cache()
            val count = scores.count
            val sumScores = scores.map(_._2).reduce(_ + _)
            val positiveCount = scores.filter(_._1 > 0).count
            val positiveSumScores = scores.filter(_._1 > 0).map(_._2).reduce(_ + _)
            val negativeCount = count - positiveCount
            val negativeSumScores = sumScores - positiveSumScores
            scores.unpersist()
            (sumScores / count, positiveSumScores / positiveCount,
             negativeSumScores / negativeCount, sumScores)
        }

        // Part 1 - Compute auPRC
        val trainPredictionAndLabels = train.sample(true, 0.1).map(t =>
            (SplitterNode.getScore(0, localNodes, t._2).toDouble -
                SplitterNode.getScore(0, localNodes, t._2, lastResample).toDouble, t._1.toDouble)
        ).cache()

        val testPredictionAndLabels = test.map(t =>
            (SplitterNode.getScore(0, localNodes, t._2).toDouble -
                SplitterNode.getScore(0, localNodes, t._2, lastResample).toDouble, t._1.toDouble)
        ).cache()

        val testRefPredictionAndLabels = testRef.map(t =>
            (SplitterNode.getScore(0, localNodes, t._2).toDouble, t._1.toDouble)
        ).cache()

        val trainMetrics = new BinaryClassificationMetrics(trainPredictionAndLabels)
        val auPRCTrain = trainMetrics.areaUnderPR + adjust(trainMetrics.pr.take(2))
        val lossFuncTrain = getLossFunc(trainPredictionAndLabels)
        val testMetrics = new BinaryClassificationMetrics(testPredictionAndLabels)
        val auPRCTest = testMetrics.areaUnderPR + adjust(testMetrics.pr.take(2))
        val lossFuncTest = getLossFunc(testPredictionAndLabels)

        var testRefMetrics = testMetrics
        var auPRCTestRef = auPRCTest
        var lossFuncTestRef = lossFuncTest
        if (test.id != testRef.id) {
            if (iteration % 1 == 0) {
                testRefMetrics = new BinaryClassificationMetrics(testRefPredictionAndLabels)
                auPRCTestRef = testRefMetrics.areaUnderPR + adjust(testRefMetrics.pr.take(2))
                lossFuncTestRef = getLossFunc(testRefPredictionAndLabels)
            } else {
                auPRCTestRef = Double.NaN
                lossFuncTestRef = (Double.NaN, Double.NaN, Double.NaN, Double.NaN)
            }
        }

        println("Training auPRC = " + auPRCTrain)
        println("Training average score = " + lossFuncTrain._1)
        println("Training average score (positive) = " + lossFuncTrain._2)
        println("Training average score (negative) = " + lossFuncTrain._3)
        println("Training scores = " + lossFuncTrain._4)
        println("Testing auPRC = " + auPRCTest)
        println("Testing average score = " + lossFuncTest._1)
        println("Testing average score (positive) = " + lossFuncTest._2)
        println("Testing average score (negative) = " + lossFuncTest._3)
        println("Testing (ref) auPRC = " + auPRCTestRef)
        println("Testing (ref) average score = " + lossFuncTestRef._1)
        println("Testing (ref) average score (positive) = " + lossFuncTestRef._2)
        println("Testing (ref) average score (negative) = " + lossFuncTestRef._3)
        // if (iteration % 1000 == 0) {
        //     println("Training PR = " + trainMetrics.pr.collect.toList)
        //     println("Testing PR = " + testMetrics.pr.collect.toList)
        //     println("Testing (ref) PR = " + testRefMetrics.pr.collect.toList)
        // }

        // Part 2 - Compute effective counts
        // TODO: these need not to be done repeatedly
        val trainCount = train.count
        val positiveTrainCount = train.filter(_._1 > 0).count
        val negativeTrainCount = trainCount - positiveTrainCount

        val wSum = glomTrain.map(_._3.sum).sum
        val wsqSum = glomTrain.map(_._3.map(t => t * t).sum).sum
        val effectiveCount = (wSum * wSum / wsqSum) / trainCount

        // val wPositive = weights.value.zip(y.value).filter(_._2 > 0).map(_._1)
        // val wSumPositive = wPositive.reduce(_ + _)
        // val wsqSumPositive = wPositive.map(s => s * s).reduce(_ + _)
        // val effectiveCountPositive = (wSumPositive * wSumPositive / wsqSumPositive) / positiveTrainCount
        // val wSumNegative = wSum - wSumPositive
        // val wsqSumNegative = wsqSum - wsqSumPositive
        // val effectiveCountNegative = (wSumNegative * wSumNegative / wsqSumNegative) / negativeTrainCount

        println("Effective count = " + effectiveCount)
        // println("Positive effective count = " + effectiveCountPositive)
        // println("Negative effective count = " + effectiveCountNegative)
        trainPredictionAndLabels.unpersist()
        testPredictionAndLabels.unpersist()
        testRefPredictionAndLabels.unpersist()
        (lossFuncTrain._1, lossFuncTest._1)
    }
}
