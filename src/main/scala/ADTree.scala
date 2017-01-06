package sparkboost

import collection.mutable.ListBuffer

class SplitterNode(val index: Int, val cond: Condition,
                   val prtValidCheck: (Vector[Double] => Boolean),
                   val onLeft: Boolean) extends java.io.Serializable {
    var leftPredict = 0.0
    var rightPredict = 0.0
    val leftChild = ListBuffer[Int]()
    val rightChild = ListBuffer[Int]()

    def validCheck(instance: Vector[Double]) = {
        prtValidCheck(instance)
    }

    def check(instance: Vector[Double], preChecked: Boolean = true) = {
        if (!preChecked && !validCheck(instance)) {
            0
        } else {
            cond.check(instance)
        }
    }

    def predict(instance: Vector[Double], preChecked: Boolean = true) = {
        if (!preChecked && !validCheck(instance)) {
            0.0
        } else if (cond.check(instance) > 0) {
            leftPredict
        }
        rightPredict
    }

    def setPredict(predict1: Double, predict2: Double) {
        leftPredict = predict1
        rightPredict = predict2
    }

    def addChild(onLeft: Boolean, childIndex: Int) {
        if (onLeft) {
            leftChild.append(childIndex)
        } else {
            rightChild.append(childIndex)
        }
    }

    override def toString() = {
        "Node " + index + ": " + cond + " (" + leftPredict + ", " + rightPredict + "), " +
        "position on " + (if (onLeft) "left" else "right") + ", " +
        "number of childs " + leftChild.size + "/" + rightChild.size
    }
}

object SplitterNode {
    def apply(index: Int, cond: Condition,
              prtValidCheck: (Vector[Double] => Boolean), onLeft: Boolean) = {
        new SplitterNode(index, cond, prtValidCheck, onLeft)
    }

    def getScore(curIndex: Int, nodes: List[SplitterNode], instance: Vector[Double],
                 maxIndex: Int = 0): Double = {
        if (maxIndex > 0 && curIndex >= maxIndex) {
            0.0
        } else {
            var score = 0.0
            val node = nodes(curIndex)
            node.check(instance) match {
                case 1 => {
                    score += node.leftPredict
                    for (c <- node.leftChild) {
                        score += getScore(c, nodes, instance, maxIndex)
                    }
                }
                case -1 => {
                    score += node.rightPredict
                    for (c <- node.rightChild) {
                        score += getScore(c, nodes, instance, maxIndex)
                    }
                }
                case _ => {}
            }
            score
        }
    }
}
