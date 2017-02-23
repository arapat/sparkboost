package sparkboost

class Instance(val y: Int, val X: Array[Double], var w: Double,
               var scores: Array[Int]) extends java.io.Serializable {
    def setWeight(weight: Double) {
        w = weight
    }

    override def toString() = {
        s"Instance($y, X, $w, $scores)"
    }
}

object Instance {
    def apply(y: Int, X: Array[Double], w: Double = 1.0,
              scores: Array[Int] = Array[Int]()) = {
        new Instance(y, X, w, scores)
    }

    def setScores(inst: Instance, nodes: Array[SplitterNode]) {
        inst.scores = Array[Int]()
        nodes.foreach {t =>
            inst.scores :+= t.check(inst)
        }
    }
}
