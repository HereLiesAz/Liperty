package com.hereliesaz.liperty.voicebox.cloning

import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import kotlin.math.PI
import kotlin.math.sin

@RunWith(RobolectricTestRunner::class)
class AudioPreprocessorTest {

    private val preprocessor = AudioPreprocessor(RuntimeEnvironment.getApplication())

    // ── VAD Tests ────────────────────────────────────────────────────────

    @Test
    fun testVadSilenceProducesNoRegions() {
        val silence = FloatArray(16000) // 1 second of silence
        val regions = preprocessor.detectSpeechRegions(silence)
        assertTrue("Silence should produce no speech regions", regions.isEmpty())
    }

    @Test
    fun testVadDetectsSineWaveAsSpeech() {
        // Generate a 440 Hz sine wave at moderate amplitude (~speech energy)
        val sampleRate = 16000
        val durationSamples = sampleRate * 2 // 2 seconds
        val signal = FloatArray(durationSamples) { i ->
            (0.3f * sin(2.0 * PI * 440.0 * i / sampleRate)).toFloat()
        }
        val regions = preprocessor.detectSpeechRegions(signal)
        assertTrue("Sine wave should be detected as speech", regions.isNotEmpty())

        // The region should cover most of the signal
        val totalSpeech = regions.sumOf { it.second - it.first }
        assertTrue(
            "Speech region should cover >80% of signal ($totalSpeech / $durationSamples)",
            totalSpeech > durationSamples * 0.8
        )
    }

    @Test
    fun testVadDetectsSpeechBetweenSilence() {
        val sr = 16000
        // 1s silence + 2s speech + 1s silence
        val signal = FloatArray(sr * 4)
        for (i in sr until sr * 3) {
            signal[i] = (0.3f * sin(2.0 * PI * 300.0 * i / sr)).toFloat()
        }

        val regions = preprocessor.detectSpeechRegions(signal)
        assertTrue("Should detect at least one speech region", regions.isNotEmpty())

        // First region should start around sample 16000 (1s mark)
        val firstStart = regions.first().first
        assertTrue("Speech should start near 1s mark (got $firstStart)", firstStart in (sr - 1024)..(sr + 1024))
    }

    @Test
    fun testVadEmptyInput() {
        val regions = preprocessor.detectSpeechRegions(floatArrayOf())
        assertTrue(regions.isEmpty())
    }

    @Test
    fun testVadVeryShortInput() {
        val regions = preprocessor.detectSpeechRegions(FloatArray(100))
        assertTrue(regions.isEmpty())
    }

    // ── Segment Refinement Tests ────────────────────────────────────────

    @Test
    fun testRefineMergesCloseSegments() {
        // Two segments 200 samples apart (< 300ms merge gap at 16kHz = 4800 samples)
        val regions = listOf(0 to 20000, 20200 to 40000)
        val refined = preprocessor.refineSpeechSegments(regions, 50000, mergeGapSamples = 4800)

        assertEquals("Close segments should merge into one", 1, refined.size)
        assertEquals(0, refined[0].first)
        assertEquals(40000, refined[0].second)
    }

    @Test
    fun testRefineKeepsDistantSegments() {
        // Two segments 10000 samples apart (> 4800 merge gap)
        val regions = listOf(0 to 20000, 30000 to 50000)
        val refined = preprocessor.refineSpeechSegments(regions, 60000, mergeGapSamples = 4800)
        assertEquals("Distant segments should remain separate", 2, refined.size)
    }

    @Test
    fun testRefineDiscardsShortSegments() {
        // Segment of 8000 samples = 0.5s (below 1s minimum at 16kHz = 16000)
        val regions = listOf(0 to 8000)
        val refined = preprocessor.refineSpeechSegments(regions, 10000, minSamples = 16000)
        assertTrue("Short segment should be discarded", refined.isEmpty())
    }

    @Test
    fun testRefineSplitsLongSegments() {
        // 25 seconds at 16kHz = 400000 samples (above 10s max = 160000)
        val regions = listOf(0 to 400000)
        val refined = preprocessor.refineSpeechSegments(regions, 400000, maxSamples = 160000)

        assertTrue("Long segment should be split into multiple", refined.size >= 2)
        for ((start, end) in refined) {
            assertTrue("Each split should be <= max", end - start <= 160000)
        }
    }

    @Test
    fun testRefineEmptyInput() {
        val refined = preprocessor.refineSpeechSegments(emptyList(), 0)
        assertTrue(refined.isEmpty())
    }

    // ── Resampling Tests ────────────────────────────────────────────────

    @Test
    fun testResampleSameRate() {
        val input = floatArrayOf(1f, 2f, 3f, 4f)
        val output = preprocessor.resample(input, 16000, 16000)
        assertArrayEquals(input, output, 0.001f)
    }

    @Test
    fun testResample44100to16000() {
        val input = FloatArray(44100) { it.toFloat() / 44100 }
        val output = preprocessor.resample(input, 44100, 16000)

        val expectedLen = (44100.0 / 44100 * 16000).toInt()
        // Allow 1 sample tolerance
        assertTrue(
            "Output length should be ~$expectedLen (got ${output.size})",
            output.size in (expectedLen - 1)..(expectedLen + 1)
        )
    }

    @Test
    fun testResample48000to16000() {
        val input = FloatArray(48000) { it.toFloat() / 48000 }
        val output = preprocessor.resample(input, 48000, 16000)

        // 48000 / 3 = 16000
        val expectedLen = 16000
        assertTrue(
            "48kHz → 16kHz should produce ~$expectedLen samples (got ${output.size})",
            output.size in (expectedLen - 1)..(expectedLen + 1)
        )
    }

    @Test
    fun testResamplePreservesAmplitude() {
        // DC signal should maintain its value
        val input = FloatArray(44100) { 0.5f }
        val output = preprocessor.resample(input, 44100, 16000)
        for (s in output) {
            assertEquals("DC value should be preserved", 0.5f, s, 0.01f)
        }
    }

    @Test
    fun testResampleEmptyInput() {
        val output = preprocessor.resample(floatArrayOf(), 44100, 16000)
        assertTrue(output.isEmpty())
    }

    // ── Integration-style Tests ─────────────────────────────────────────

    @Test
    fun testEndToEndSyntheticSpeech() {
        val sr = 16000
        // 0.5s silence + 3s "speech" + 0.5s silence + 2s "speech" + 1s silence
        val totalSamples = sr * 7
        val signal = FloatArray(totalSamples)

        // First "speech" block: 0.5s–3.5s
        for (i in (sr / 2) until (sr * 7 / 2)) {
            signal[i] = (0.2f * sin(2.0 * PI * 250.0 * i / sr)).toFloat()
        }
        // Second "speech" block: 4s–6s
        for (i in (sr * 4) until (sr * 6)) {
            signal[i] = (0.15f * sin(2.0 * PI * 350.0 * i / sr)).toFloat()
        }

        val regions = preprocessor.detectSpeechRegions(signal)
        val refined = preprocessor.refineSpeechSegments(regions, totalSamples)

        assertTrue("Should detect at least 2 speech regions", refined.size >= 2)

        // All segments should be within valid duration bounds
        for ((start, end) in refined) {
            val durMs = ((end - start).toLong() * 1000) / sr
            assertTrue("Segment duration ${durMs}ms should be >= 1000ms", durMs >= 1000)
            assertTrue("Segment duration ${durMs}ms should be <= 10000ms", durMs <= 10000)
        }
    }
}
