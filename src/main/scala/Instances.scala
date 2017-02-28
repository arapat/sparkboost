package sparkboost

import org.apache.spark.mllib.linalg.SparseVector

class Instances(val x: SparseVector, val ptr: Array[Int],
                val index: Int, val splits: Array[Double]) extends java.io.Serializable {
}

object Instances {
    def createSlices(sliceFrac: Double, x: Array[Double]) = {
        // assume values are sorted
        val sliceSize = math.max(1, (x.size * sliceFrac).floor.toInt)
        (sliceSize until x.size by sliceSize).map(
            idx => 0.5 * (x(idx - 1) + x(idx))
        ).distinct.toArray :+ Double.MaxValue
    }

    def apply(x: SparseVector, ptr: Array[Int], index: Int, sliceFrac: Double) = {
        val slices = createSlices(sliceFrac, x.toDense.values)
        new Instances(x, ptr, index, slices)
    }
}
