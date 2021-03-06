package sparkboost

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector

class Instances(val batchId: Int, val x: SparseVector, val ptr: Array[Int],
                val index: Int, val splits: Array[Double],
                val active: Boolean) extends java.io.Serializable {
    // set active=true if the current index of this group of instances is being used for training
    val xVec = (new DenseVector((x.toDense.values).zip(ptr).sortBy(_._2).map(_._1))).toSparse
}

object Instances {
    def createSlices(numSlices: Int, x: Array[Double]) = {
        // assume values are sorted
        var distinct = Array[Double](x(0))
        x.foreach(ix => if (ix != distinct.head) distinct = ix +: distinct)
        val sliceSize = math.max(1, (distinct.size.toDouble / numSlices).floor.toInt)
        (sliceSize until distinct.size by sliceSize).map(
            idx => 0.5 * (distinct(idx - 1) + distinct(idx))
        ).toArray :+ Double.MaxValue
    }

    def apply(batchId: Int, x: SparseVector, ptr: Array[Int],
              index: Int, numSlices: Int, active: Boolean) = {
        val slices = createSlices(numSlices, x.toDense.values)
        new Instances(batchId, x, ptr, index, slices, active)
    }
}
