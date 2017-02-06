package sparkboost

import collection.mutable.ArrayBuffer

class Instance(val y: Int, val X: Vector[Double], var w: Double,
               var scores: ArrayBuffer[Int]) extends java.io.Serializable {
    def appendScore(score: Int) {
        scores.append(score)
    }

    def setScores(nodes: List[SplitterNode]) {
        scores = ArrayBuffer[Int]()
        for (node <- nodes) {
            scores.append(node.check(this))
        }
    }

    def setWeight(weight: Double) {
        w = weight
    }

    override def toString() = {
        s"Instance($y, X, $w, $scores)"
    }
}

object Instance {
    def apply(y: Int, X: Vector[Double], w: Double = 1.0,
              scores: ArrayBuffer[Int] = ArrayBuffer[Int]()) = {
        new Instance(y, X, w, scores)
    }
}
