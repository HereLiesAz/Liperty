package com.hereliesaz.liperty.voicebox.cloning

import org.junit.Assert.*
import org.junit.Test

class SpeakerClustererTest {

    private val clusterer = SpeakerClusterer()

    // ── Cosine Similarity ───────────────────────────────────────────────

    @Test
    fun testCosineSimilarityIdentical() {
        val a = floatArrayOf(1f, 2f, 3f, 4f)
        assertEquals(1.0f, clusterer.cosineSimilarity(a, a), 0.001f)
    }

    @Test
    fun testCosineSimilarityOrthogonal() {
        val a = floatArrayOf(1f, 0f, 0f)
        val b = floatArrayOf(0f, 1f, 0f)
        assertEquals(0.0f, clusterer.cosineSimilarity(a, b), 0.001f)
    }

    @Test
    fun testCosineSimilarityOpposite() {
        val a = floatArrayOf(1f, 2f, 3f)
        val b = floatArrayOf(-1f, -2f, -3f)
        assertEquals(-1.0f, clusterer.cosineSimilarity(a, b), 0.001f)
    }

    @Test
    fun testCosineSimilarityScaledVectors() {
        val a = floatArrayOf(1f, 2f, 3f)
        val b = floatArrayOf(10f, 20f, 30f)
        assertEquals(1.0f, clusterer.cosineSimilarity(a, b), 0.001f)
    }

    // ── Centroid ────────────────────────────────────────────────────────

    @Test
    fun testComputeCentroid() {
        val embeddings = listOf(
            floatArrayOf(1f, 2f, 3f),
            floatArrayOf(3f, 4f, 5f)
        )
        val centroid = clusterer.computeCentroid(embeddings)
        assertArrayEquals(floatArrayOf(2f, 3f, 4f), centroid, 0.001f)
    }

    @Test
    fun testComputeCentroidSingle() {
        val emb = floatArrayOf(5f, 10f, 15f)
        val centroid = clusterer.computeCentroid(listOf(emb))
        assertArrayEquals(emb, centroid, 0.001f)
    }

    // ── Clustering ──────────────────────────────────────────────────────

    private fun makeSegment(durationMs: Long = 2000, snr: Float = 15f) =
        AudioPreprocessor.SpeechSegment(
            pcmData = FloatArray(16000 * durationMs.toInt() / 1000),
            startTimeMs = 0,
            durationMs = durationMs,
            sourceFileName = "test.wav",
            rmsEnergy = 0.1f,
            snrEstimate = snr
        )

    @Test
    fun testSingleSegmentProducesOneCluster() {
        val segments = listOf(makeSegment())
        val embeddings = listOf(floatArrayOf(1f, 0f, 0f, 0f))

        val result = clusterer.cluster(segments, embeddings)
        assertTrue(result.singleSpeaker)
        assertEquals(1, result.clusters.size)
        assertEquals(1, result.clusters[0].segments.size)
    }

    @Test
    fun testSimilarEmbeddingsClusterTogether() {
        val segments = List(4) { makeSegment() }
        // All very similar (same speaker)
        val embeddings = listOf(
            floatArrayOf(1f, 0.1f, 0f, 0f),
            floatArrayOf(1f, 0.05f, 0.02f, 0f),
            floatArrayOf(1f, 0.08f, 0.01f, 0f),
            floatArrayOf(1f, 0.12f, 0f, 0.01f)
        )

        val result = clusterer.cluster(segments, embeddings)
        assertTrue("All similar embeddings should be one cluster", result.singleSpeaker)
        assertEquals(1, result.clusters.size)
        assertEquals(4, result.clusters[0].segments.size)
    }

    @Test
    fun testTwoDistinctSpeakers() {
        val segments = List(6) { makeSegment() }
        // Speaker A: pointing in +x direction
        // Speaker B: pointing in +y direction
        val embeddings = listOf(
            floatArrayOf(1f, 0f, 0f, 0f),     // A
            floatArrayOf(1f, 0.1f, 0f, 0f),   // A
            floatArrayOf(1f, 0.05f, 0f, 0f),  // A
            floatArrayOf(0f, 1f, 0f, 0f),     // B
            floatArrayOf(0.1f, 1f, 0f, 0f),   // B
            floatArrayOf(0f, 1f, 0.05f, 0f)   // B
        )

        val result = clusterer.cluster(segments, embeddings)
        assertFalse("Should detect two speakers", result.singleSpeaker)
        assertEquals(2, result.clusters.size)

        // Each cluster should have 3 segments
        val sizes = result.clusters.map { it.segments.size }.sorted()
        assertEquals(listOf(3, 3), sizes)
    }

    @Test
    fun testEmptyInput() {
        val result = clusterer.cluster(emptyList(), emptyList())
        assertTrue(result.singleSpeaker)
        assertTrue(result.clusters.isEmpty())
    }

    @Test
    fun testRepresentativeSegmentPreference() {
        // The representative should favor longer duration and higher SNR
        val short = makeSegment(durationMs = 1000, snr = 10f)
        val long = makeSegment(durationMs = 5000, snr = 20f)
        val segments = listOf(short, long)
        val embeddings = listOf(
            floatArrayOf(1f, 0f, 0f),
            floatArrayOf(1f, 0.1f, 0f)
        )

        val result = clusterer.cluster(segments, embeddings)
        assertEquals(1, result.clusters.size)
        assertEquals(
            "Representative should be the longer, higher-SNR segment",
            5000L,
            result.clusters[0].representativeSegment.durationMs
        )
    }

    @Test
    fun testClusterTotalDuration() {
        val segments = listOf(
            makeSegment(durationMs = 2000),
            makeSegment(durationMs = 3000)
        )
        val embeddings = listOf(
            floatArrayOf(1f, 0f, 0f),
            floatArrayOf(1f, 0.05f, 0f)
        )

        val result = clusterer.cluster(segments, embeddings)
        assertEquals(5000L, result.clusters[0].totalSpeechDurationMs)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testMismatchedInputSizes() {
        clusterer.cluster(
            listOf(makeSegment()),
            listOf(floatArrayOf(1f), floatArrayOf(2f))
        )
    }
}
