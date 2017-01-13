package sparkboost

import collection.mutable.ListBuffer

class SplitterNode(val index: Int, val cond: Condition, val prtIndex: Int,
                   val validChecks: List[(Condition, Int)],
                   val onLeft: Boolean) extends java.io.Serializable {
    var leftPredict = 0.0
    var rightPredict = 0.0
    val leftChild = ListBuffer[Int]()
    val rightChild = ListBuffer[Int]()

    def check(instance: Vector[Double], preChecked: Boolean = true): Int = {
        if (!preChecked) {
            for (cr <- validChecks) {
                if (cr._1.check(instance) != cr._2) {
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
              validChecks: List[(Condition, Int)], onLeft: Boolean) = {
        new SplitterNode(index, cond, prtIndex, validChecks, onLeft)
    }

    def getScore(curIndex: Int, nodes: List[SplitterNode], instance: Vector[Double],
                 maxIndex: Int = 0): Double = {
        if (maxIndex > 0 && curIndex >= maxIndex) {
            0.0
        } else {
            val node = nodes(curIndex)
            node.check(instance) match {
                case 1 => {
                    if (node.leftChild.nonEmpty) {
                        node.leftChild.map(getScore(_, nodes, instance, maxIndex)).reduce(_ + _)
                    } else {
                        0.0
                    } + node.leftPredict
                }
                case -1 => {
                    if (node.rightChild.nonEmpty) {
                        node.rightChild.map(getScore(_, nodes, instance, maxIndex)).reduce(_ + _)
                    } else {
                        0.0
                    } + node.rightPredict
                }
            }
        }
    }
}
