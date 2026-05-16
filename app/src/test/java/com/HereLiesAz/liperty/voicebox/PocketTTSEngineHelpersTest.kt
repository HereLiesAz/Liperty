package com.hereliesaz.liperty.voicebox

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/**
 * Unit tests for the pure-Kotlin helpers inside [PocketTTSEngine].
 *
 * Covers:
 * - [PocketTTSEngine.linearResample] — naive resampler used to
 *   normalize arbitrary-rate reference audio to 24 kHz before the
 *   SE extractor.
 * - [PocketTTSEngine.averageEmbeddings] — multi-clip embedding
 *   centroid, the "more samples = more like you" lever.
 *
 * ONNX-dependent paths (init, generateAudio, extractEmbeddingFromPcm)
 * are exercised by integration tests on-device with real models.
 *
 * Robolectric `@Config(sdk=[34])` works around `targetSdk=37`
 * confusing the default RobolectricTestRunner — see CLAUDE.md.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34], manifest = Config.NONE)
class PocketTTSEngineHelpersTest {

    private lateinit var engine: PocketTTSEngine

    @Before
    fun setUp() {
        engine = PocketTTSEngine(ApplicationProvider.getApplicationContext())
        // Don't call initialize() — we're testing pure helpers, no
        // ONNX sessions needed.
    }

    // ── linearResample ──────────────────────────────────────────────

    @Test
    fun `linearResample identity case returns input unchanged`() {
        val input = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        val out = engine.linearResample(input, 24_000, 24_000)
        assertEquals(input.size, out.size)
        for (i in input.indices) assertEquals(input[i], out[i], 1e-6f)
    }

    @Test
    fun `linearResample 2x upsample doubles output length`() {
        val input = floatArrayOf(0f, 1f, 0f, 1f)
        val out = engine.linearResample(input, 12_000, 24_000)
        assertEquals(input.size * 2, out.size)
    }

    @Test
    fun `linearResample 0_5x downsample halves output length`() {
        val input = FloatArray(100) { it.toFloat() }
        val out = engine.linearResample(input, 48_000, 24_000)
        assertEquals(50, out.size)
        // Endpoints should be preserved (modulo linear interp).
        assertEquals(input[0], out[0], 1e-3f)
    }

    @Test
    fun `linearResample empty input returns empty`() {
        val out = engine.linearResample(FloatArray(0), 16_000, 24_000)
        assertEquals(0, out.size)
    }

    @Test
    fun `linearResample interpolates between samples`() {
        // 2-sample ramp upsampled by 4x — middle value should be
        // somewhere between the endpoints (linear blend).
        val input = floatArrayOf(0f, 1f)
        val out = engine.linearResample(input, 6_000, 24_000)
        // Length should be 8 (2 * 4).
        assertEquals(8, out.size)
        assertEquals(0f, out[0], 1e-6f)
        // Final sample should be at or close to 1.
        assertTrue("Last sample should be > 0.7, got ${out[out.size - 1]}",
            out[out.size - 1] > 0.7f)
        // Middle samples should be monotonically increasing.
        for (i in 1 until out.size) {
            assertTrue("Resampled ramp not monotonic at $i", out[i] >= out[i - 1])
        }
    }

    // ── averageEmbeddings ───────────────────────────────────────────

    @Test
    fun `averageEmbeddings single clip returns same vector`() {
        val emb = FloatArray(PocketTTSEngine.EMBEDDING_DIM) { it.toFloat() }
        val averaged = engine.averageEmbeddings(listOf(emb))
        assertEquals(emb.size, averaged.size)
        for (i in emb.indices) assertEquals(emb[i], averaged[i], 1e-6f)
    }

    @Test
    fun `averageEmbeddings two clips returns midpoint`() {
        val a = FloatArray(PocketTTSEngine.EMBEDDING_DIM) { 0f }
        val b = FloatArray(PocketTTSEngine.EMBEDDING_DIM) { 1f }
        val averaged = engine.averageEmbeddings(listOf(a, b))
        for (v in averaged) assertEquals(0.5f, v, 1e-6f)
    }

    @Test
    fun `averageEmbeddings many clips converges toward centroid`() {
        // Simulate noisy estimates of a true centroid at 0.7.
        // The average across N clips should approach 0.7 as N grows.
        val truth = 0.7f
        val rng = java.util.Random(42)
        val clips = List(30) {
            FloatArray(PocketTTSEngine.EMBEDDING_DIM) {
                truth + (rng.nextFloat() - 0.5f) * 0.2f
            }
        }
        val averaged = engine.averageEmbeddings(clips)
        // Each dim should be within 0.03 of truth (loose: 30 noisy
        // samples, uniform noise of amplitude 0.1, expected stderr
        // ~0.018). 0.03 gives us multi-sigma room.
        for (v in averaged) assertEquals(truth, v, 0.03f)
    }

    @Test
    fun `averageEmbeddings throws on dimension mismatch`() {
        val a = FloatArray(256)
        val b = FloatArray(192) // legacy ECAPA dim — must not silently average
        assertThrows(IllegalArgumentException::class.java) {
            engine.averageEmbeddings(listOf(a, b))
        }
    }

    @Test
    fun `averageEmbeddings throws on empty list`() {
        assertThrows(IllegalArgumentException::class.java) {
            engine.averageEmbeddings(emptyList())
        }
    }

    // ── EMBEDDING_DIM contract ──────────────────────────────────────

    @Test
    fun `EMBEDDING_DIM matches OpenVoice v2 contract`() {
        // Documented contract: SE extractor produces (1, 256, 1) float
        // tensors. If this constant ever changes, the persisted profile
        // format and the tone converter's expected shape both need to
        // change in lockstep.
        assertEquals(256, PocketTTSEngine.EMBEDDING_DIM)
    }

    @Test
    fun `TTS_OUTPUT_SAMPLE_RATE_HZ matches OpenVoice v2 contract`() {
        // OpenVoice v2 emits 24 kHz audio. AudioTrack / AudioRouter
        // are wired to this constant — wrong value plays back at the
        // wrong pitch/speed (e.g. 22050 would play 8% slow).
        assertEquals(24_000, PocketTTSEngine.TTS_OUTPUT_SAMPLE_RATE_HZ)
        assertEquals(
            PocketTTSEngine.TTS_OUTPUT_SAMPLE_RATE_HZ,
            PocketTTSEngine.SE_EXTRACTOR_SAMPLE_RATE_HZ
        )
    }

    @Test
    fun `isReady is false before initialize`() {
        // No ONNX sessions loaded → cloning not ready.
        assertEquals(false, engine.isReady)
        assertEquals(false, engine.canSynthesize)
    }
}
