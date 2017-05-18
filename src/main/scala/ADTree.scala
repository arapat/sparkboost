package sparkboost

import java.io._
import math.max

import sparkboost.utils.Comparison

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.Vector

class SplitterNode(val index: Int, val prtIndex: Int, val depth: Int,
                   val splitIndex: Int, val splitVal: Double, val splitEval: Boolean)
                        extends java.io.Serializable with Comparison {
    var pred = 0.0
    var child = List[Int]()

    def check(value: Double, fIndex: Int, inNode: Boolean) = {
        require(inNode == true)
        require(fIndex == splitIndex)
        splitIndex < 0 || ((compare(value, splitVal) <= 0) == splitEval)
    }

    def predict(value: Double, fIndex: Int, inNode: Boolean) = {
        if (check(value, fIndex, inNode)) {
            pred
        } else {
            0.0
        }
    }

    def setPredict(value: Double) {
        pred = value
    }

    def addChild(childIndex: Int) {
        child = childIndex +: child
    }

    override def toString() = {
        val nChild = child.size
        val position = if (prtIndex >= 0) {
            s", positioned under the node $prtIndex,"
        } else ""

        s"Node $index: Index $splitIndex <= $splitVal == $splitEval (predict $pred)" + position +
        s" has $nChild children, depth $depth."
    }
}

object SplitterNode {
    def apply(index: Int, prtIndex: Int, depth: Int,
                splitPoint: (Int, Double, Boolean)) = {
        new SplitterNode(index, prtIndex, depth,
                splitPoint._1, splitPoint._2, splitPoint._3)
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
                 maxNumNodes: Int = -1): Double = {
        if (maxNumNodes >= 0 && curIndex >= maxNumNodes || curIndex >= nodes.size) {
            0.0
        } else {
            val node = nodes(curIndex)
            if (node.check(instance(max(0, node.splitIndex)), node.splitIndex, true)) {
                node.pred + (
                    if (node.child.nonEmpty) {
                        node.child.map(t => getScore(t, nodes, instance, maxNumNodes))
                                  .reduce(_ + _)
                    } else {
                        0.0
                    }
                )
            } else {
                0.0
            }
        }
    }
}
