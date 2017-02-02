package sparkboost

import java.io._

import collection.mutable.ListBuffer

class SplitterNode(val index: Int, val cond: Condition, val prtIndex: Int,
                   val onLeft: Boolean) extends java.io.Serializable {
    var leftPredict = 0.0
    var rightPredict = 0.0
    val leftChild = ListBuffer[Int]()
    val rightChild = ListBuffer[Int]()

    def check(instance: Instance, preChecked: Boolean = false) = {
        if (preChecked || prtIndex < 0 ||
                (onLeft && instance.scores(prtIndex) > 0) ||
                (!onLeft && instance.scores(prtIndex) < 0)) {
            cond.check(instance.X)
        } else {
            0
        }
    }

    def predict(instance: Instance, preChecked: Boolean = false) = {
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
        val nLeftChild = leftChild.size
        val nRightChild = rightChild.size
        val position = if (prtIndex >= 0) {
            ", positioned under the " + (if (onLeft) "left" else "right") +
            s" side of the node $prtIndex,"
        } else ""

        s"Node $index: $cond ($leftPredict, $rightPredict)" + position +
        s" has $nLeftChild left and $nRightChild right children."
    }
}

object SplitterNode {
    def apply(index: Int, cond: Condition, prtIndex: Int, onLeft: Boolean) = {
        new SplitterNode(index, cond, prtIndex, onLeft)
    }

    def save(nodes: List[SplitterNode], filepath: String) {
        val oos = new ObjectOutputStream(new FileOutputStream(filepath))
        oos.writeObject(nodes)
        oos.close()
    }

    def load(filepath: String) = {
        class NodesInputStream(f: FileInputStream) extends ObjectInputStream(f) {
            override def resolveClass(desc: java.io.ObjectStreamClass): Class[_] = {
                try {
                    Class.forName(desc.getName, false, getClass.getClassLoader)
                } catch {
                    case ex: ClassNotFoundException => super.resolveClass(desc)
                }
            }
        }

        val ois = new NodesInputStream(new FileInputStream(filepath))
        val nodes = ois.readObject.asInstanceOf[List[SplitterNode]]
        ois.close
        nodes
    }

    def getScore(curIndex: Int, nodes: List[SplitterNode], instance: Instance,
                 maxIndex: Int = 0): Double = {
        if (maxIndex > 0 && curIndex >= maxIndex) {
            0.0
        } else {
            val node = nodes(curIndex)
            node.check(instance, preChecked=true) match {
                case 1 => {
                    node.leftPredict + (
                        if (node.leftChild.nonEmpty) {
                            node.leftChild.map(t => getScore(t, nodes, instance, maxIndex))
                                          .reduce(_ + _)
                        } else {
                            0.0
                        }
                    )
                }
                case -1 => {
                    node.rightPredict + (
                        if (node.rightChild.nonEmpty) {
                            node.rightChild.map(t => getScore(t, nodes, instance, maxIndex))
                                           .reduce(_ + _)
                        } else {
                            0.0
                        }
                    )
                }
            }
        }
    }
}
