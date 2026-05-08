package com.hereliesaz.liperty.voicebox.cloning

import kotlin.math.ln
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Builds and incrementally updates [VoiceProfile]s.
 *
 * Supports:
 * - Creating a new profile from initial segments + embeddings
 * - Adding more samples to an existing profile (the core "add more samples" feature)
 * - Quality scoring based on multiple signals
 * - Duration- and SNR-weighted embedding averaging
 */
class VoiceProfileBuilder {

    companion object {
        /** Diminishing-returns target: 20 samples gives full score on the count dimension. */
        private const val TARGET_SAMPLE_COUNT = 20
        /** 30 seconds of clean speech is the quality target. */
        private const val TARGET_DURATION_SEC = 30.0
        /** 20 dB SNR is considered clean speech. */
        private const val TARGET_SNR_DB = 20.0
    }

    /**
     * Creates a new [VoiceProfile] from initial speech segments and their embeddings.
     */
    fun createProfile(
        name: String,
        segments: List<AudioPreprocessor.SpeechSegment>,
        embeddings: List<FloatArray>
    ): VoiceProfile {
        require(segments.size == embeddings.size)
        require(segments.isNotEmpty()) { "Cannot create profile from empty segments" }

        val now = System.currentTimeMillis()
        val samples = segments.zip(embeddings).map { (seg, emb) ->
            VoiceSample(
                sourceFileName = seg.sourceFileName,
                durationMs = seg.durationMs,
                embedding = emb,
                snrEstimate = seg.snrEstimate,
                addedAtMs = now
            )
        }

        val averaged = weightedAverageEmbedding(samples)
        val quality = computeQualityScore(samples)

        return VoiceProfile(
            name = name,
            createdAtMs = now,
            lastUpdatedMs = now,
            samples = samples,
            averagedEmbedding = averaged,
            qualityScore = quality
        )
    }

    /**
     * Adds new samples to an existing profile and recomputes the embedding.
     */
    fun addSamples(
        existing: VoiceProfile,
        newSegments: List<AudioPreprocessor.SpeechSegment>,
        newEmbeddings: List<FloatArray>
    ): VoiceProfile {
        require(newSegments.size == newEmbeddings.size)
        if (newSegments.isEmpty()) return existing

        val now = System.currentTimeMillis()
        val newSamples = newSegments.zip(newEmbeddings).map { (seg, emb) ->
            VoiceSample(
                sourceFileName = seg.sourceFileName,
                durationMs = seg.durationMs,
                embedding = emb,
                snrEstimate = seg.snrEstimate,
                addedAtMs = now
            )
        }

        val allSamples = existing.samples + newSamples
        val averaged = weightedAverageEmbedding(allSamples)
        val quality = computeQualityScore(allSamples)

        return existing.copy(
            lastUpdatedMs = now,
            samples = allSamples,
            averagedEmbedding = averaged,
            qualityScore = quality
        )
    }

    /**
     * Computes a quality score in [0, 1] based on four signals:
     *
     * | Signal                  | Weight | Full score when            |
     * |-------------------------|--------|----------------------------|
     * | Sample count            | 0.20   | >= 20 samples              |
     * | Total speech duration   | 0.30   | >= 30 seconds              |
     * | Embedding consistency   | 0.30   | Low intra-profile variance |
     * | Average SNR             | 0.20   | >= 20 dB                   |
     */
    fun computeQualityScore(samples: List<VoiceSample>): Float {
        if (samples.isEmpty()) return 0f

        val countScore = min(1.0, samples.size.toDouble() / TARGET_SAMPLE_COUNT) * 0.20
        val durationSec = samples.sumOf { it.durationMs } / 1000.0
        val durationScore = min(1.0, durationSec / TARGET_DURATION_SEC) * 0.30

        val consistencyScore = if (samples.size >= 2) {
            val variance = embeddingVariance(samples.map { it.embedding })
            // Typical intra-speaker variance with ECAPA-TDNN is ~0.05-0.15
            // Normalize so variance=0 → 1.0, variance=0.5 → 0.0
            val normalized = min(1.0, (variance / 0.5).toDouble())
            (1.0 - normalized) * 0.30
        } else {
            0.15 // Single sample: half of maximum consistency score
        }

        val avgSnr = samples.map { it.snrEstimate.toDouble() }.average()
        val snrScore = min(1.0, avgSnr / TARGET_SNR_DB) * 0.20

        return (countScore + durationScore + consistencyScore + snrScore).toFloat().coerceIn(0f, 1f)
    }

    /**
     * Weighted average of embeddings.
     * Weight = duration_normalized * snr_normalized
     * Longer, cleaner segments contribute more to the final voice identity.
     */
    fun weightedAverageEmbedding(samples: List<VoiceSample>): FloatArray {
        if (samples.isEmpty()) return floatArrayOf()

        val dim = samples[0].embedding.size

        // Compute weights: duration * SNR (both are positive)
        val weights = samples.map { s ->
            val durationWeight = s.durationMs.toFloat() / 1000f // seconds
            val snrWeight = (s.snrEstimate.coerceAtLeast(1f)) // at least 1 dB
            durationWeight * snrWeight
        }
        val totalWeight = weights.sum().coerceAtLeast(1e-10f)

        val result = FloatArray(dim)
        for ((idx, sample) in samples.withIndex()) {
            val w = weights[idx] / totalWeight
            for (i in result.indices) {
                result[i] += sample.embedding[i] * w
            }
        }
        return result
    }

    /**
     * Intra-profile embedding variance (average squared distance from centroid).
     * Lower = more consistent speaker identity across samples.
     */
    fun embeddingVariance(embeddings: List<FloatArray>): Float {
        if (embeddings.size < 2) return 0f

        val dim = embeddings[0].size
        // Centroid
        val mean = FloatArray(dim)
        for (emb in embeddings) {
            for (i in mean.indices) mean[i] += emb[i]
        }
        for (i in mean.indices) mean[i] /= embeddings.size

        // Average squared distance
        var totalDist = 0.0
        for (emb in embeddings) {
            var dist = 0.0
            for (i in emb.indices) {
                val d = emb[i] - mean[i]
                dist += d * d
            }
            totalDist += dist
        }
        return (totalDist / embeddings.size).toFloat()
    }
}
