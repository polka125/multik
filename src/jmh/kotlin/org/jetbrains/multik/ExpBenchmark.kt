package org.jetbrains.multik

import org.jetbrains.multik.api.JvmMath
import org.jetbrains.multik.api.d2array
import org.jetbrains.multik.api.mk
import org.jetbrains.multik.core.D2
import org.jetbrains.multik.core.D2Array
import org.jetbrains.multik.core.Ndarray
import org.jetbrains.multik.core.get
import org.jetbrains.multik.jni.NativeMath
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.TimeUnit
import kotlin.math.exp
import kotlin.random.Random

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 5)
@Measurement(iterations = 5)
@Fork(value = 2, jvmArgsPrepend = ["-Djava.library.path=./build/libs"])
open class ExpBenchmark {

    @Param("10", "100", "1000")
    var size: Int = 0
    private lateinit var arg: D2Array<Double>
    private lateinit var result: Ndarray<Double, D2>
    private lateinit var ran: Random

    @Setup
    fun generate() {
        ran = Random(1)
        arg = mk.d2array(size, size) { ran.nextDouble() }
    }

    @TearDown
    fun assertResult() {
        for (i in 0 until size) {
            for (j in 0 until size) {
                val expected = exp(arg[i, j])
                if (expected != result[i, j]) throw IllegalStateException("Exponential function calculation error: actual - exp(${arg[i, j]}) = ${result[i, j]}, expected - $expected")
            }
        }
    }

    @Benchmark
    fun expJniBench(bh: Blackhole) {
        result = NativeMath.exp(arg)
        bh.consume(result)
    }

    @Benchmark
    fun expJvmBench(bh: Blackhole) {
        result = JvmMath.exp(arg)
        bh.consume(result)
    }
}