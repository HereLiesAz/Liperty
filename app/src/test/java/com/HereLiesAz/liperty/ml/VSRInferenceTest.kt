package com.hereliesaz.liperty.ml

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

        override fun initialize(): Boolean {
            initialized = true
            return true
        }

        override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
            runCalled = true
            // Fill output buffer with dummy probabilities
            // Shape [1, 5, 40] (Batch=1, Time=5, Vocab=40)
            // Let's output "AA", "AE", "AH", "AO", "AW"
            // AA=1, AE=2, AH=3, AO=4, AW=5
            val vocabSize = 39
            val timeSteps = 16

            // For each time step, set prob of corresponding index to 1.0
            for (t in 0 until timeSteps) {
                for (v in 0 until vocabSize) {
                    val prob = if (v == (t % vocabSize) + 1) 1.0f else 0.0f
                    outputBuffer.putFloat(prob)
                }
            }
        }

        override fun getOutputShape(index: Int): IntArray = intArrayOf(1, 16, 39)

        override fun getInputShape(index: Int): IntArray = intArrayOf(1, 16, 224, 224, 3)

        override fun close() {}
    }

    @Test
    fun testInferenceWithFakeEngine() {
        val engine = FakeModelEngine()
        val vsr = VSRInference(engine)

        vsr.initialize()
        assert(engine.initialized)

        val frames = List(5) { Bitmap.createBitmap(128, 64, Bitmap.Config.ARGB_8888) }
        val result = vsr.runInference(frames)

        assert(engine.runCalled)
        // AA=1, AE=2, AH=3, AO=4, AW=5, AY=6, B=7, CH=8, D=9, DH=10, EH=11, ER=12, EY=13, F=14, G=15, HH=16
        assertEquals("AAAEAHAWAYDDHEREYG", result.text)
    }

    @Test
    fun testInferenceHandlesException() {
         val engine = object : ModelEngine {
            override fun initialize(): Boolean = true
             override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
                 throw RuntimeException("Inference crashed")
             }
             override fun getOutputShape(outputIndex: Int): IntArray {
                 return intArrayOf(1, 16, 39)
             }
             override fun getInputShape(inputIndex: Int): IntArray {
                 return intArrayOf(1, 16, 224, 224, 3)
             }
             override fun close() {}
         }

         val vsr = VSRInference(engine)
         val frames = listOf(Bitmap.createBitmap(128, 64, Bitmap.Config.ARGB_8888))
         val result = vsr.runInference(frames)

         assertEquals("", result.text)
    }
}
