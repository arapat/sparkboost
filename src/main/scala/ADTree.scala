package sparkboost

import java.io._

import sparkboost.utils.Comparison

import org.apache.spark.mllib.linalg.Vector

class SplitterNode(val index: Int, val prtIndex: Int, val onLeft: Boolean,
                   val splitIndex: Int, val splitVal: Double)
                        extends java.io.Serializable with Comparison {
    var leftPredict = 0.0
    var rightPredict = 0.0
    var leftChild = List[Int]()
    var rightChild = List[Int]()

    def check(value: Double, fIndex: Int, inNode: Boolean) = {
        require(inNode == true)
        require(fIndex == splitIndex)
        if (splitIndex < 0 || compare(instance.X(splitIndex), splitVal) <= 0) {
            -1
        } else {
            1
        }
    }

    def predict(value: Double, fIndex: Int, inNode: Boolean) = {
        check(value, fIndex, inNode) match {
            case -1 => leftPredict
            case 1 => rightPredict
        }
    }

    def setPredict(predict1: Double, predict2: Double) {
        leftPredict = predict1
        rightPredict = predict2
    }

    def addChild(onLeft: Boolean, childIndex: Int) {
        if (onLeft) {
            leftChild = childIndex +: leftChild
        } else {
            rightChild = childIndex +: rightChild
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
    def apply(index: Int, prtIndex: Int, onLeft: Boolean, splitPoint: (Int, Double)) = {
        new SplitterNode(index, prtIndex, onLeft, splitPoint._1, splitPoint._2)
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

    def getScore(curIndex: Int, nodes: Array[SplitterNode], instance: Vector,
                 maxNumNodes: Int = 0): Double = {
        if (maxNumNodes > 0 && curIndex >= maxNumNodes) {
            0.0
        } else {
            val node = nodes(curIndex)
            node.check(instance(node.splitIndex), node.splitIndex, true) match {
                case -1 => {
                    node.leftPredict + (
                        if (node.leftChild.nonEmpty) {
                            node.leftChild.map(t => getScore(t, nodes, instance, maxNumNodes))
                                          .reduce(_ + _)
                        } else {
                            0.0
                        }
                    )
                }
                case 1 => {
                    node.rightPredict + (
                        if (node.rightChild.nonEmpty) {
                            node.rightChild.map(t => getScore(t, nodes, instance, maxNumNodes))
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
