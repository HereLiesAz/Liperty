package com.hereliesaz.liperty.ml

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Tests for the parameterized [VSRInference] surface used by the Auto-AVSR
 * backend: per-channel pixel normalization (mean/std) and a pluggable decode
 * function.
 *
 * The Auto-AVSR ONNX expects `(pixel/255 - 0.421) / 0.165` grayscale, not the
 * default `pixel/255` the existing VideoMAE path uses. The decode swap is
 * needed because Auto-AVSR emits 5049 subword logits, not 40 ARPABET phonemes.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class VSRInferenceParamsTest {

    /**
     * Captures the input bytes the engine sees, so a test can inspect the
     * normalization the caller applied. Returns a fixed dummy output.
     */
    class CapturingEngine(
        private val inputShape: IntArray = intArrayOf(1, 1, 1, 1, 1),  // (B, T, H, W, C) NTHWC
        private val outputShape: IntArray = intArrayOf(1, 1, 1),
        private val layout: InputLayout = InputLayout.NTHWC,
    ) : ModelEngine {
        var capturedFloats: FloatArray? = null
        override fun initialize(): Boolean = true
        override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
            inputBuffer.rewind()
            val floats = FloatArray(inputBuffer.capacity() / 4)
            inputBuffer.asFloatBuffer().get(floats)
            capturedFloats = floats
            outputBuffer.rewind()
            for (i in 0 until outputBuffer.capacity() / 4) {
                outputBuffer.putFloat(0f)
            }
        }
        override fun getInputShape(inputIndex: Int): IntArray =
            if (inputIndex == 0) inputShape else intArrayOf()
        override fun getOutputShape(outputIndex: Int): IntArray =
            if (outputIndex == 0) outputShape else intArrayOf()
        override fun getInputLayout(): InputLayout = layout
        override fun close() {}
    }

    private fun whitePixel(): Bitmap {
        val bm = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        bm.setPixel(0, 0, Color.WHITE)
        return bm
    }

    private fun greyPixel(value: Int): Bitmap {
        val bm = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        bm.setPixel(0, 0, Color.rgb(value, value, value))
        return bm
    }

    @Test
    fun defaultsToPixelDividedBy255() {
        val engine = CapturingEngine()
        val vsr = VSRInference(engine)
        vsr.runInference(listOf(whitePixel()))
        // 255/255 = 1.0
        assertNotNull(engine.capturedFloats)
        assertEquals(1.0f, engine.capturedFloats!![0], 1e-5f)
    }

    @Test
    fun appliesAutoAvsrMeanAndStd() {
        val engine = CapturingEngine()
        val vsr = VSRInference(engine, pixelMean = 0.421f, pixelStd = 0.165f)
        vsr.runInference(listOf(whitePixel()))
        // (255/255 - 0.421) / 0.165 = (1.0 - 0.421) / 0.165 = 3.50909...
        val expected = (1.0f - 0.421f) / 0.165f
        assertEquals(expected, engine.capturedFloats!![0], 1e-4f)
    }

    @Test
    fun appliesNormalizationToMidGreyPixel() {
        val engine = CapturingEngine()
        val vsr = VSRInference(engine, pixelMean = 0.421f, pixelStd = 0.165f)
        vsr.runInference(listOf(greyPixel(128)))
        // (128/255 - 0.421) / 0.165
        val expected = (128f / 255f - 0.421f) / 0.165f
        assertEquals(expected, engine.capturedFloats!![0], 1e-4f)
    }

    @Test
    fun customDecoderReceivesProbabilitiesAndItsReturnIsTheResultText() {
        // Use 5-vocab so the parity is easy to inspect.
        val engine = CapturingEngine(outputShape = intArrayOf(1, 1, 5))
        var seenProbsShape: IntArray? = null
        val vsr = VSRInference(
            engine,
            customDecoder = { probs ->
                seenProbsShape = intArrayOf(probs.size, probs[0].size)
                "from-custom-decoder"
            },
        )
        val result = vsr.runInference(listOf(whitePixel()))
        assertEquals("from-custom-decoder", result.text)
        // (T_out=1, V=5) — probabilities post-softmax, not raw logits.
        assertArrayEquals(intArrayOf(1, 5), seenProbsShape)
    }

    @Test
    fun whenCustomDecoderReturnsBlankResultTextIsBlank() {
        val engine = CapturingEngine(outputShape = intArrayOf(1, 1, 5))
        val vsr = VSRInference(engine, customDecoder = { "" })
        val result = vsr.runInference(listOf(whitePixel()))
        assertEquals("", result.text)
    }
}
