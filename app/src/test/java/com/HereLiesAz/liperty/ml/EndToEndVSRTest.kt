package com.HereLiesAz.liperty.ml

import android.graphics.Bitmap
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.nio.ByteBuffer

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class EndToEndVSRTest {

    @Test
    fun testEndToEndPipeline() {
        // Mock Engine
        val mockEngine = object : ModelEngine {
            override fun initialize() {}
            override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
                // Fill output with zeros or dummy probs
                val outputSize = outputBuffer.capacity() / 4
                for (i in 0 until outputSize) {
                    outputBuffer.putFloat(0.0f)
                }
            }
            // Batch=1, Time=5, Vocab=28
            override fun getOutputShape(outputIndex: Int) = intArrayOf(1, 5, 28)
            override fun close() {}
        }

        val vsr = VSRInference(mockEngine)
        vsr.initialize()

        // Create dummy frames
        val frames = List(5) { Bitmap.createBitmap(88, 88, Bitmap.Config.ARGB_8888) }

        // Run Inference
        val result = vsr.runInference(frames)

        assertNotNull(result)
        // Since probs are 0, decoding might be empty string, but result object should exist
        assertNotNull(result.text)
    }
}
