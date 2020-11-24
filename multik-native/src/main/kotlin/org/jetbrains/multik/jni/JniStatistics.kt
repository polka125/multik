package org.jetbrains.multik.jni

import org.jetbrains.multik.api.Statistics
import org.jetbrains.multik.ndarray.data.*

public object NativeStatistics : Statistics {
    override fun <T : Number, D : Dimension> median(a: MultiArray<T, D>): Double? {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> average(a: MultiArray<T, D>, weights: MultiArray<T, D>?): Double {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension> mean(a: MultiArray<T, D>): Double {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dimension, O : Dimension> mean(a: MultiArray<T, D>, axis: Int): Ndarray<Double, O> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> meanD2(a: MultiArray<T, D2>, axis: Int): Ndarray<Double, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> meanD3(a: MultiArray<T, D3>, axis: Int): Ndarray<Double, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> meanD4(a: MultiArray<T, D4>, axis: Int): Ndarray<Double, D3> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> meanDN(a: MultiArray<T, DN>, axis: Int): Ndarray<Double, D4> {
        TODO("Not yet implemented")
    }
}