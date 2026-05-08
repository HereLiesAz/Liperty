package com.hereliesaz.liperty.voicebox.cloning

import org.junit.Assert.*
import org.junit.Test

class VoiceProfileBuilderTest {

    private val builder = VoiceProfileBuilder()

    private fun makeSegment(durationMs: Long = 2000, snr: Float = 15f, fileName: String = "test.wav") =
        AudioPreprocessor.SpeechSegment(
            pcmData = FloatArray(16 * durationMs.toInt()), // dummy
            startTimeMs = 0,
            durationMs = durationMs,
            sourceFileName = fileName,
            rmsEnergy = 0.1f,
            snrEstimate = snr
        )

    // ── Create Profile ──────────────────────────────────────────────────

    @Test
    fun testCreateProfileSingle() {
        val segments = listOf(makeSegment())
        val embeddings = listOf(floatArrayOf(1f, 2f, 3f))

        val profile = builder.createProfile("test", segments, embeddings)

        assertEquals("test", profile.name)
        assertEquals(1, profile.sampleCount)
        assertEquals(2000L, profile.totalSpeechDurationMs)
        assertArrayEquals(floatArrayOf(1f, 2f, 3f), profile.averagedEmbedding, 0.01f)
        assertTrue(profile.qualityScore in 0f..1f)
    }

    @Test
    fun testCreateProfileMultiple() {
        val segments = listOf(
            makeSegment(3000, 20f),
            makeSegment(2000, 10f)
        )
        val embeddings = listOf(
            floatArrayOf(1f, 0f, 0f),
            floatArrayOf(0f, 1f, 0f)
        )

        val profile = builder.createProfile("test", segments, embeddings)
        assertEquals(2, profile.sampleCount)
        assertEquals(5000L, profile.totalSpeechDurationMs)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testCreateProfileEmpty() {
        builder.createProfile("test", emptyList(), emptyList())
    }

    // ── Add Samples ─────────────────────────────────────────────────────

    @Test
    fun testAddSamplesIncrements() {
        val initial = builder.createProfile(
            "test",
            listOf(makeSegment()),
            listOf(floatArrayOf(1f, 0f, 0f))
        )

        val updated = builder.addSamples(
            initial,
            listOf(makeSegment(3000)),
            listOf(floatArrayOf(0f, 1f, 0f))
        )

        assertEquals(2, updated.sampleCount)
        assertEquals(5000L, updated.totalSpeechDurationMs)
        assertTrue(updated.lastUpdatedMs >= initial.lastUpdatedMs)
    }

    @Test
    fun testAddEmptySamplesReturnsSame() {
        val initial = builder.createProfile(
            "test",
            listOf(makeSegment()),
            listOf(floatArrayOf(1f, 0f, 0f))
        )

        val same = builder.addSamples(initial, emptyList(), emptyList())
        assertEquals(initial.sampleCount, same.sampleCount)
    }

    // ── Quality Score ───────────────────────────────────────────────────

    @Test
    fun testQualityScoreIncreasesWithMoreSamples() {
        val fewSamples = (1..3).map {
            VoiceSample("id$it", "test.wav", 2000, floatArrayOf(1f, 0f, 0f), 15f)
        }
        val manySamples = (1..15).map {
            VoiceSample("id$it", "test.wav", 2000, floatArrayOf(1f, 0f, 0f), 15f)
        }

        val scoreFew = builder.computeQualityScore(fewSamples)
        val scoreMany = builder.computeQualityScore(manySamples)
        assertTrue("More samples should yield higher quality ($scoreMany > $scoreFew)", scoreMany > scoreFew)
    }

    @Test
    fun testQualityScoreAlwaysInRange() {
        // Empty
        assertEquals(0f, builder.computeQualityScore(emptyList()), 0.001f)

        // Single sample
        val single = listOf(VoiceSample("1", "t.wav", 1000, floatArrayOf(1f, 0f), 5f))
        val s = builder.computeQualityScore(single)
        assertTrue("Score $s should be in [0, 1]", s in 0f..1f)

        // Many high-quality samples
        val many = (1..30).map {
            VoiceSample("$it", "t.wav", 2000, floatArrayOf(1f, 0f), 25f)
        }
        val m = builder.computeQualityScore(many)
        assertTrue("Score $m should be in [0, 1]", m in 0f..1f)
    }

    @Test
    fun testQualityScoreHigherSnrBetter() {
        val lowSnr = (1..5).map {
            VoiceSample("$it", "t.wav", 3000, floatArrayOf(1f, 0f, 0f), 3f)
        }
        val highSnr = (1..5).map {
            VoiceSample("$it", "t.wav", 3000, floatArrayOf(1f, 0f, 0f), 25f)
        }

        val scoreLow = builder.computeQualityScore(lowSnr)
        val scoreHigh = builder.computeQualityScore(highSnr)
        assertTrue("Higher SNR should yield better quality", scoreHigh > scoreLow)
    }

    // ── Weighted Average ────────────────────────────────────────────────

    @Test
    fun testWeightedAverageFavorsLongerSegments() {
        val samples = listOf(
            VoiceSample("a", "t.wav", 10000, floatArrayOf(1f, 0f), 15f), // 10s, high weight
            VoiceSample("b", "t.wav", 1000, floatArrayOf(0f, 1f), 15f)   // 1s, low weight
        )

        val avg = builder.weightedAverageEmbedding(samples)
        // First dimension should dominate since the 10s segment has 10x the duration weight
        assertTrue(
            "Longer segment should dominate (${avg[0]} > ${avg[1]})",
            avg[0] > avg[1]
        )
    }

    @Test
    fun testWeightedAverageFavorsHigherSnr() {
        val samples = listOf(
            VoiceSample("a", "t.wav", 2000, floatArrayOf(1f, 0f), 30f),  // high SNR
            VoiceSample("b", "t.wav", 2000, floatArrayOf(0f, 1f), 3f)    // low SNR
        )

        val avg = builder.weightedAverageEmbedding(samples)
        assertTrue(
            "Higher SNR segment should dominate (${avg[0]} > ${avg[1]})",
            avg[0] > avg[1]
        )
    }

    @Test
    fun testWeightedAverageSingleSample() {
        val samples = listOf(
            VoiceSample("a", "t.wav", 2000, floatArrayOf(3f, 7f, 11f), 15f)
        )
        val avg = builder.weightedAverageEmbedding(samples)
        assertArrayEquals(floatArrayOf(3f, 7f, 11f), avg, 0.001f)
    }

    // ── Embedding Variance ──────────────────────────────────────────────

    @Test
    fun testVarianceZeroForIdentical() {
        val embeddings = List(5) { floatArrayOf(1f, 2f, 3f) }
        assertEquals(0f, builder.embeddingVariance(embeddings), 0.001f)
    }

    @Test
    fun testVariancePositiveForDifferent() {
        val embeddings = listOf(
            floatArrayOf(1f, 0f),
            floatArrayOf(0f, 1f)
        )
        assertTrue(builder.embeddingVariance(embeddings) > 0f)
    }

    @Test
    fun testVarianceSingleSample() {
        assertEquals(0f, builder.embeddingVariance(listOf(floatArrayOf(1f, 2f))), 0.001f)
    }
}
