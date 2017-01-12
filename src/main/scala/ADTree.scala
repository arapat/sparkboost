package sparkboost

import collection.mutable.ListBuffer

class SplitterNode(val index: Int, val cond: Condition, val prtIndex: Int,
                   val validChecks: List[Vector[Double] => Boolean],
                   val onLeft: Boolean) extends java.io.Serializable {
    var leftPredict = 0.0
    var rightPredict = 0.0
    val leftChild = ListBuffer[Int]()
    val rightChild = ListBuffer[Int]()

    def check(instance: Vector[Double], preChecked: Boolean = true): Int = {
        if (!preChecked) {
            for (f <- validChecks) {
                if (!f(instance)) {
                    return 0
                }
            }
        }
        cond.check(instance)
    }

    def predict(instance: Vector[Double], preChecked: Boolean = true) = {
        check(instance, preChecked) match {
            case 0 => 0
            case 1 => leftPredict
            case -1 => rightPredict
        }
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
        "position under the " + (if (onLeft) "left" else "right") + " side of the node " +
        prtIndex + ", " + "number of childs (left) " + leftChild.size +
        " (right) " + rightChild.size
    }
}

object SplitterNode {
    def apply(index: Int, cond: Condition, prtIndex: Int,
              validChecks: List[Vector[Double] => Boolean], onLeft: Boolean) = {
        new SplitterNode(index, cond, prtIndex, validChecks, onLeft)
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
