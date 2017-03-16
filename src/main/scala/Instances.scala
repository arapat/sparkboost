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
    def createSlices(sliceFrac: Double, x: Array[Double]) = {
        // assume values are sorted
        val sliceSize = math.max(1, (x.size * sliceFrac).floor.toInt)
        (sliceSize until x.size by sliceSize).map(
            idx => 0.5 * (x(idx - 1) + x(idx))
        ).distinct.toArray :+ Double.MaxValue
    }

    def apply(batchId: Int, x: SparseVector, ptr: Array[Int],
              index: Int, sliceFrac: Double, active: Boolean) = {
        val slices = createSlices(sliceFrac, x.toDense.values)
        new Instances(batchId, x, ptr, index, slices, active)
    }
}
