package sparkboost

import java.io._

import sparkboost.utils.Comparison

class SplitterNode(val index: Int, val splitIndex: Int, val splitVal: Double,
                   val prtIndex: Int, val onLeft: Boolean) extends java.io.Serializable with Comparison {
    var leftPredict = 0.0
    var rightPredict = 0.0
    var leftChild = Array[Int]()
    var rightChild = Array[Int]()

    def check(instance: Instance, preChecked: Boolean = false) = {
        if (preChecked || prtIndex < 0 ||
                (onLeft && instance.scores(prtIndex) > 0) ||
                (!onLeft && instance.scores(prtIndex) < 0)) {
            if (splitIndex < 0 || compare(instance.X(splitIndex), splitVal) <= 0) {
                1
            } else {
                -1
            }
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
            leftChild :+= childIndex
        } else {
            rightChild :+= childIndex
        }
    }

    override def toString() = {
        val nLeftChild = leftChild.size
        val nRightChild = rightChild.size
        val position = if (prtIndex >= 0) {
            ", positioned under the " + (if (onLeft) "left" else "right") +
            s" side of the node $prtIndex,"
        } else ""

        s"Node $index: Index $index <= $splitVal ($leftPredict, $rightPredict)" + position +
        s" has $nLeftChild left and $nRightChild right children."
    }
}

object SplitterNode {
    def apply(index: Int, splitIndex: Int, splitVal: Double, prtIndex: Int, onLeft: Boolean) = {
        new SplitterNode(index, splitIndex, splitVal, prtIndex, onLeft)
    }

    def save(nodes: Array[SplitterNode], filepath: String) {
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
        val nodes = ois.readObject.asInstanceOf[Array[SplitterNode]]
        ois.close
        nodes
    }

    def getScore(curIndex: Int, nodes: Array[SplitterNode], instance: Instance,
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
