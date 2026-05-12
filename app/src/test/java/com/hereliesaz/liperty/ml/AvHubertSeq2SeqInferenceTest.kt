package com.hereliesaz.liperty.ml

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/**
 * Integration test for [AvHubertSeq2SeqInference]. The encoder and
 * decoder are passed in via the [EncoderSession] and [DecoderSession]
 * interfaces, so the orchestrator's wiring (preprocessing -> encode ->
 * autoregressive decode -> detokenize) can be verified end-to-end
 * without an actual ONNX file or an Application context.
 *
 * Uses AndroidJUnit4 + sdk=34 only because the preprocessing calls
 * `Bitmap.createScaledBitmap`. No `ApplicationProvider` access -> no
 * `LipertyApplication.onCreate` trigger -> no Bluetooth/MediaPipe init
 * crash in unit-test land.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class AvHubertSeq2SeqInferenceTest {

    /** Dict the orchestrator uses to detokenize. Indices 0..3 are fairseq's
     *  canonical specials; 4+ are word pieces. */
    private val dict = listOf(
        "<s>", "<pad>", "</s>", "<unk>",
        "▁hello", "▁world",
    )

    private class FakeEncoder(
        /** What runEncode returns, regardless of input. */
        private val out: Pair<FloatArray, LongArray>,
    ) : EncoderSession {
        var encodeCalls = 0
        var lastInputShape: LongArray? = null
        override fun initialize(): Boolean = true
        override fun runEncode(input: FloatArray, inputShape: LongArray): Pair<FloatArray, LongArray> {
            encodeCalls++
            lastInputShape = inputShape
            return out
        }
        override fun close() {}
    }

    private class FakeDecoder(
        /** Token-by-token plan: each runStep returns one-hot logits picking
         *  the next token. After the plan runs out, emits the eos token (2). */
        private val plan: IntArray,
        override val vocabSize: Int = 6,
    ) : DecoderSession {
        var stepCalls = 0
        val prevTokensSeen = ArrayList<IntArray>()
        override fun initialize(): Boolean = true
        override fun runStep(
            prevTokens: IntArray,
            encoderFeatures: FloatArray,
            encShape: LongArray,
        ): FloatArray {
            prevTokensSeen.add(prevTokens.copyOf())
            val pick = if (stepCalls < plan.size) plan[stepCalls] else 2
            stepCalls++
            return FloatArray(vocabSize).also { it[pick] = 1f }
        }
        override fun close() {}
    }

    @Test
    fun decodesFakeEncoderFeaturesIntoCorrectText() {
        val featShape = longArrayOf(1L, 4L, 8L)
        val feats = FloatArray((featShape[0] * featShape[1] * featShape[2]).toInt())
        val fakeEnc = FakeEncoder(feats to featShape)
        // Decoder emits [▁hello, ▁world, </s>] -> "hello world".
        val fakeDec = FakeDecoder(plan = intArrayOf(4, 5, 2))

        val orch = AvHubertSeq2SeqInference(
            encoder = fakeEnc, decoder = fakeDec, dict = dict,
            bosId = 0, eosId = 2,
            numFrames = 4, cropSize = 8,
            pixelMean = 0f, pixelStd = 1f,
            maxDecodeSteps = 10,
        )
        assertTrue(orch.initialize())

        val frames = List(4) { Bitmap.createBitmap(8, 8, Bitmap.Config.ARGB_8888) }
        val result = orch.runInference(frames)

        assertNotNull(result)
        assertEquals("hello world", result.text)
        assertEquals(1, fakeEnc.encodeCalls)
        assertEquals(3, fakeDec.stepCalls)
        assertEquals(1f, result.confidence)
        assertTrue("processing time non-negative", result.processingTimeMs >= 0L)
    }

    @Test
    fun handsEncoderTheNcthwShapeFromConfig() {
        val featShape = longArrayOf(1L, 2L, 8L)
        val feats = FloatArray((featShape[0] * featShape[1] * featShape[2]).toInt())
        val fakeEnc = FakeEncoder(feats to featShape)
        val fakeDec = FakeDecoder(plan = intArrayOf(2))

        val orch = AvHubertSeq2SeqInference(
            encoder = fakeEnc, decoder = fakeDec, dict = dict,
            numFrames = 50, cropSize = 88,
        )
        orch.initialize()
        orch.runInference(List(50) { Bitmap.createBitmap(88, 88, Bitmap.Config.ARGB_8888) })

        val seen = fakeEnc.lastInputShape!!
        assertEquals(5, seen.size)
        assertEquals(longArrayOf(1L, 1L, 50L, 88L, 88L).toList(), seen.toList())
    }

    @Test
    fun growsPrevTokensOneAtATimeStartingAtBos() {
        val featShape = longArrayOf(1L, 4L, 8L)
        val fakeEnc = FakeEncoder(FloatArray(32) to featShape)
        val fakeDec = FakeDecoder(plan = intArrayOf(4, 5, 4, 2))

        val orch = AvHubertSeq2SeqInference(
            encoder = fakeEnc, decoder = fakeDec, dict = dict,
            numFrames = 4, cropSize = 8,
            bosId = 0, eosId = 2, maxDecodeSteps = 10,
        )
        orch.initialize()
        orch.runInference(List(4) { Bitmap.createBitmap(8, 8, Bitmap.Config.ARGB_8888) })

        assertEquals(4, fakeDec.stepCalls)
        assertEquals(listOf(0), fakeDec.prevTokensSeen[0].toList())
        assertEquals(listOf(0, 4), fakeDec.prevTokensSeen[1].toList())
        assertEquals(listOf(0, 4, 5), fakeDec.prevTokensSeen[2].toList())
        assertEquals(listOf(0, 4, 5, 4), fakeDec.prevTokensSeen[3].toList())
    }

    @Test
    fun handsEncoderTheNtchwShapeWhenLayoutOverridden() {
        // SyncVSR's bundled E2E expects NTCHW (batch, time, channels=1,
        // H, W) — not NCTHW. The orchestrator must declare that on the
        // ONNX input tensor or the encoder errors out at shape-check.
        val featShape = longArrayOf(1L, 2L, 8L)
        val feats = FloatArray((featShape[0] * featShape[1] * featShape[2]).toInt())
        val fakeEnc = FakeEncoder(feats to featShape)
        val fakeDec = FakeDecoder(plan = intArrayOf(2))

        val orch = AvHubertSeq2SeqInference(
            encoder = fakeEnc, decoder = fakeDec, dict = dict,
            numFrames = 16, cropSize = 88,
            inputLayout = InputLayout.NTCHW,
        )
        orch.initialize()
        orch.runInference(List(16) { Bitmap.createBitmap(88, 88, Bitmap.Config.ARGB_8888) })

        val seen = fakeEnc.lastInputShape!!
        assertEquals(5, seen.size)
        assertEquals(longArrayOf(1L, 16L, 1L, 88L, 88L).toList(), seen.toList())
    }

    @Test
    fun emptyDecodeOutputsEmptyText() {
        val featShape = longArrayOf(1L, 2L, 8L)
        val fakeEnc = FakeEncoder(FloatArray(16) to featShape)
        val fakeDec = FakeDecoder(plan = intArrayOf(2))

        val orch = AvHubertSeq2SeqInference(
            encoder = fakeEnc, decoder = fakeDec, dict = dict,
            numFrames = 4, cropSize = 8, maxDecodeSteps = 10,
        )
        orch.initialize()
        val result = orch.runInference(List(4) { Bitmap.createBitmap(8, 8, Bitmap.Config.ARGB_8888) })

        assertEquals("", result.text)
    }
}
