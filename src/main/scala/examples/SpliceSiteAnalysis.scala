// package sparkboost.examples
// 
// import org.apache.spark.SparkConf
// import org.apache.spark.SparkContext
// import org.apache.spark.rdd.RDD
// import org.apache.spark.mllib.linalg.SparseVector
// import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
// 
// import sparkboost._
// 
// object SpliceSiteAnalysis {
//     /*
//     Commandline options:
// 
//         --load-model    - File path to load the model
//         --type          - Operation type:
//                             1 -> get testing PR
//                             2 -> get model description
// 
//     Optional:
//         --test          - file path to the test data (required for "--type 1")
//         --max-node      - maximum number of nodes to use in the loaded model
//     */
//     type TestRDDType = RDD[(Int, SparseVector)]
//     val nodeName = InstanceFactory.indexMap.map(_.swap)
// 
//     def parseOptions(options: Array[String]) = {
//         options.zip(options.slice(1, options.size))
//                .zip(0 until options.size).filter(_._2 % 2 == 0).map(_._1)
//                .map {case (key, value) => (key.slice(2, key.size), value)}
//                .toMap
//     }
// 
//     def printStats(test: TestRDDType, nodes: Array[SplitterNode]) {
//         // manual fix the auPRC computation bug in MLlib
//         def adjust(points: Array[(Double, Double)]) = {
//             require(points.length == 2)
//             require(points.head == (0.0, 1.0))
//             val y = points.last
//             y._1 * (y._2 - 1.0) / 2.0
//         }
// 
//         // Part 1 - Compute auPRC
//         val predictionAndLabels = test.map {case t =>
//             (SplitterNode.getScore(0, nodes, t._2).toDouble, t._1.toDouble)
//         }.cache()
// 
//         val metrics = new BinaryClassificationMetrics(predictionAndLabels)
//         val auPRC = metrics.areaUnderPR + adjust(metrics.pr.take(2))
// 
//         println("Testing auPRC = " + auPRC)
//         println("Testing PR = " + metrics.pr.collect.toList)
//     }
// 
//     def main(args: Array[String]) {
//         // Define SparkContext
//         val conf = new SparkConf()
//         conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//             .set("spark.kryoserializer.buffer.mb","24")
//         // Parse and read options
//         val options = parseOptions(args)
//         val op = options("type").toInt
//         val modelReadPath = options("load-model")
//         val allNodes = SplitterNode.load(modelReadPath)
//         val maxNode = options.getOrElse("max-node", allNodes.size.toString).toInt
//         val nodes = allNodes.take(maxNode)
// 
//         def desc(index: Int): String = {
//             if (index <= 0) {
//                 ""
//             } else {
//                 val prtDesc = desc(nodes(index).prtIndex)
//                 val sel =
//                     if (1.0 <= nodes(index).splitVal == nodes(index).splitEval) {
//                         "is "
//                     } else {
//                         "isnot "
//                     }
//                 val icon = nodeName(nodes(index).splitIndex)
//                 if (prtDesc == "") {
//                     sel + icon
//                 } else {
//                     prtDesc + " and " + sel + icon
//                 }
//             }
//         }
// 
//         if (op == 1) {
//             val sc = new SparkContext(conf)
//             val testPath = options("test")
//             val data = sc.textFile(testPath).map(InstanceFactory.rowToInstance).cache()
//             println("Distinct positive samples in the training data (test data): " +
//                 data.filter(_._1 > 0).count)
//             println("Distinct negative samples in the training data (test data): " +
//                 data.filter(_._1 < 0).count)
//             printStats(data, nodes)
//             sc.stop()
//         } else {
//             nodes.sortBy(-_.pred.abs).foreach(node => {
//                 println(node.index + ", " + node.depth + ", " + node.pred + ", " + desc(node.index))
//             })
//         }
//     }
// }
