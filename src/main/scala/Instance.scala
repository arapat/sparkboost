package sparkboost

import org.apache.spark.mllib.linalg.Vector

class Instance(val y: Int, val X: Vector, var w: Double,
               val scores: Array[Int]) extends java.io.Serializable {
    def setWeight(weight: Double) {
        w = weight
    }

    override def toString() = {
        s"Instance($y, X, $w, $scores)"
    }
}

object Instance {
    def apply(y: Int, X: Vector, w: Double = 1.0,
              scores: Array[Int] = Array[Int]()) = {
        new Instance(y, X, w, scores)
    }

    def clone(inst: Instance, w: Double, nodes: Array[SplitterNode]) = {
        val offset = nodes.size - 1
        var scores = (0 to offset).map(_ => 0).toArray

        def getScores(idx: Int) {
            val node = nodes(idx)
            val c = node.check(inst, true)
            scores(offset - idx) = c
            c match {
                case 1 => nodes(idx).leftChild.map(getScores)
                case -1 => nodes(idx).rightChild.map(getScores)
                case 0 => Nil
            }
        }

        getScores(0)
        Instance(inst.y, inst.X, w, scores)
    }
}
