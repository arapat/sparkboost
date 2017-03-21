package sparkboost.utils

import sparkboost._

object ModelSlice {
    /*
    Commandline options:

    --slice         - Number of iterations for keeping
    --save-model    - File path to save the model
    --load-model    - File path to load the model
    */

    def parseOptions(options: Array[String]) = {
        options.zip(options.slice(1, options.size))
               .zip(0 until options.size).filter(_._2 % 2 == 0).map(_._1)
               .map {case (key, value) => (key.slice(2, key.size), value)}
               .toMap
    }

    def main(args: Array[String]) {
        // Parse and read options
        val options = parseOptions(args)
        val slice = options("slice").toInt
        val modelReadPath = options("load-model")
        val modelWritePath = options("save-model")

        val nodes = SplitterNode.load(modelReadPath)
        val ret = nodes.slice(0, slice).map(node => {
            val unode = SplitterNode(node.index, node.prtIndex, node.depth,
                                     (node.splitIndex, node.splitVal, node.splitEval))
            unode.setPredict(node.pred)
            unode.child = node.child.filter(_ < slice)
            unode
        })
        SplitterNode.save(ret, modelWritePath)
    }
}

// sbt "run-main sparkboost.utils.ModelSlice --load-model <model-path>
// --save-model <model-path> --slice 400"
