package com.HereLiesAz.liperty.ml

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import java.nio.ByteBuffer

@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class VSRInferenceTest {

    class FakeModelEngine : ModelEngine {
        var initialized = false
        var runCalled = false

        override fun initialize() {
            initialized = true
        }

        override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
            runCalled = true
            // Fill output buffer with dummy probabilities
            // Shape [1, 5, 28] (Batch=1, Time=5, Vocab=28)
            // Let's output "A", "B", "C", "D", "E"
            // A=1, B=2, C=3, D=4, E=5
            val vocabSize = 28
            val timeSteps = 5

            // For each time step, set prob of corresponding index to 1.0
            for (t in 0 until timeSteps) {
                for (v in 0 until vocabSize) {
                    val prob = if (v == t + 1) 1.0f else 0.0f
                    outputBuffer.putFloat(prob)
                }
            }
        }

        override fun getOutputShape(outputIndex: Int): IntArray {
            return intArrayOf(1, 5, 28)
        }

        override fun close() {}
    }

    @Test
    fun testInferenceWithFakeEngine() {
        val engine = FakeModelEngine()
        val vsr = VSRInference(engine)

        vsr.initialize()
        assert(engine.initialized)

        val frames = List(5) { Bitmap.createBitmap(88, 88, Bitmap.Config.ARGB_8888) }
        val result = vsr.runInference(frames)

        assert(engine.runCalled)
        // A, B, C, D, E -> "ABCDE"
        assertEquals("ABCDE", result.text)
    }

    @Test
    fun testInferenceHandlesException() {
         val engine = object : ModelEngine {
             override fun initialize() {}
             override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
                 throw RuntimeException("Inference crashed")
             }
             override fun getOutputShape(outputIndex: Int): IntArray {
                 return intArrayOf(1, 5, 28)
             }
             override fun close() {}
         }

         val vsr = VSRInference(engine)
         val frames = listOf(Bitmap.createBitmap(88, 88, Bitmap.Config.ARGB_8888))
         val result = vsr.runInference(frames)

         assertEquals("Model Missing", result.text)
    }
}
