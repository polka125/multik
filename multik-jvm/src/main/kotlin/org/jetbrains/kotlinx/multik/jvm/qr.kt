package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.lang.Math.min
import kotlin.math.hypot
import kotlin.math.sqrt


fun householderTransform(x: D2Array<Double>): Pair<Double, D1Array<Double>> {
    val alpha = x[0, 0]
//    val xnorm: Double = sqrt(x[1..x.shape[0], 0].map { it * it }.sum())
    var xnorm = 0.0
    for (i in 1 until x.shape[0]) {
        xnorm += x[i, 0] * x[i, 0]
    }
    xnorm = sqrt(xnorm)

    val v: D1Array<Double> = mk.empty<Double, D1>(x.shape[0])
    v[0] = 1.0
    if (xnorm == 0.0) {
        return Pair(0.0, v)
    }
    val beta = -(if(alpha >= 0) 1 else -1) * hypot(alpha, xnorm)
    val tau = (beta - alpha) / beta
    val alphaMinusBeta = alpha - beta
    for (i in 1 until v.size) {
        v[i] = x[i, 0] / alphaMinusBeta
    }
    return Pair(tau, v)
}

fun applyHousholder(x: D2Array<Double>, tau: Double, v: D1Array<Double>): D2Array<Double> {
    //x - tau * np.sum(v * x) * v
    val applied = x.deepCopy()

    for (columnNumber in 0 until x.shape[1]) {
        var scal = 0.0 // scal(x[:, columnNumber], v)
        for (i in 0 until v.size) {
            scal += v[i] * x[i, columnNumber]
        }
        for(i in 0 until v.size) {
            applied[i, columnNumber] -= tau * scal * v[i]
        }
    }
    return applied
}

fun qr(mat: D2Array<Double>): Pair<D2Array<Double>, D2Array<Double>> {

    var q = mk.identity<Double>(mat.shape[0])
    val r = mat.deepCopy()
    for (i in 0 until min(mat.shape[0],  mat.shape[1])) {
        val (tau, v) = householderTransform(r[i..r.shape[0], i..r.shape[1]] as D2Array<Double>)
        val appliedR = applyHousholder(r[i..r.shape[0], i..r.shape[1]] as D2Array<Double>, tau, v)
        for (d0 in i until r.shape[0]) {
            for (d1 in i until r.shape[1]) {
                r[d0, d1] = appliedR[d0 - i, d1 - i]
            }
        }
        q = q.transpose()
        val appliedQ = applyHousholder(q[i..q.shape[0], 0..q.shape[1]] as D2Array<Double>, tau, v)
        for (d0 in i until q.shape[0]) {
            for (d1 in 0 until q.shape[1]) {
                q[d0, d1] = appliedQ[d0 - i, d1]
            }
        }
        q = q.transpose()
    }
    return Pair(q, r)
}