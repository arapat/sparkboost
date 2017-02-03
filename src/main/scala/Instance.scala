package sparkboost

import collection.mutable.ListBuffer

class Instance(val y: Int, val X: Vector[Double], val w: Double,
               var scores: Vector[Int]) extends java.io.Serializable {
    def updateScores(nodes: List[SplitterNode]) {
        val s = ListBuffer[Int]()
        for (node <- nodes) {
            s.append(node.check(this))
            scores = s.toVector
        }
    }

    override def toString() = {
        s"Instance($y, X, $w, $scores)"
    }
}

object Instance {
    def apply(y: Int, X: Vector[Double], w: Double = 1.0,
              scores: Vector[Int] = Vector[Int]()) = {
        new Instance(y, X, w, scores)
    }
}
