package sparkboost

import scala.collection.mutable.ListBuffer

class SplitterNode(val index: Int, val cond: Condition,
                   val prtValidCheck: ((Instance, Boolean) => Boolean),
                   val onLeft: Boolean) {
    var leftPredict = 0.0
    var rightPredict = 0.0
    val leftChild = ListBuffer[Int]()
    val rightChild = ListBuffer[Int]()

    def validCheck(instance: Instance) = {
        prtValidCheck(instance, onLeft)
    }

    def check(instance: Instance, preChecked: Boolean = true) = {
        if (!preChecked && !validCheck(instance)) {
            None
        } else {
            cond.check(instance)
        }
    }

    def predict(instance: Instance, preChecked: Boolean = true) = {
        if (!preChecked && !validCheck(instance)) {
            0.0
        } else if (cond.check(instance)) {
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
}

object SplitterNode {
    def getScore(curIndex: Int, nodes: List[SplitterNode], instance: Instance,
                 maxIndex: Int = 0, quiet: Boolean = true): Double = {
        if (maxIndex > 0 && curIndex >= maxIndex) {
            0.0
        } else {
            var score = 0.0
            val node = nodes(curIndex)
            node.check(instance) match {
                case Some(true) => {
                    score += node.leftPredict
                    for (c <- node.leftChild) {
                        score += getScore(c, nodes, instance, maxIndex, quiet)
                    }
                }
                case Some(false) => {
                    score += node.rightPredict
                    for (c<- node.rightChild) {
                        score += getScore(c, nodes, instance, maxIndex, quiet)
                    }
                }
            }
            score
        }
    }
}
