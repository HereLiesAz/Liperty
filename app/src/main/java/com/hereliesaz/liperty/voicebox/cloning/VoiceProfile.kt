package com.hereliesaz.liperty.voicebox.cloning

import com.hereliesaz.liperty.voicebox.VoiceState
import java.util.UUID

/**
 * Rich voice profile supporting incremental building from multiple import sessions.
 * Contains per-sample metadata so the embedding can be recomputed when new samples
 * are added.
 */
data class VoiceProfile(
    val name: String,
    val createdAtMs: Long,
    val lastUpdatedMs: Long,
    val samples: List<VoiceSample>,
    val averagedEmbedding: FloatArray,
    /** Quality score in [0, 1] based on sample count, duration, consistency, SNR. */
    val qualityScore: Float
) {
    /** Convert to the lightweight [VoiceState] used by [com.hereliesaz.liperty.voicebox.PocketTTSEngine] at inference time. */
    fun toVoiceState(): VoiceState = VoiceState(name, averagedEmbedding)

    val totalSpeechDurationMs: Long get() = samples.sumOf { it.durationMs }
    val sampleCount: Int get() = samples.size
}

/**
 * Metadata for a single audio sample that contributed to a [VoiceProfile].
 * The raw audio is discarded after embedding extraction — only the embedding persists.
 */
data class VoiceSample(
    val id: String = UUID.randomUUID().toString(),
    val sourceFileName: String,
    val durationMs: Long,
    val embedding: FloatArray,
    val snrEstimate: Float,
    val addedAtMs: Long = System.currentTimeMillis()
)
