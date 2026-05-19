package com.hereliesaz.liperty.ml

/**
 * A single pre-processed training sample for the on-device LoRA trainer.
 *
 * Video frames are already normalized to AV-HuBERT's expected format
 * (mean=0.421, std=0.165, Rec.601 grayscale, 88x88).
 *
 * Target features are the frozen encoder's own output for this clip
 * (identity target for the MSE PoC). Future CTC variant will replace
 * targets with tokenized transcripts.
 */
data class TrainingSample(
    /** Flat float array in NCTHW layout: (1, 1, T, 88, 88). */
    val videoFrames: FloatArray,
    val numFrames: Int,
    /** Flat float array: (1, T_out, 768) — MSE target from frozen encoder. */
    val targetFeatures: FloatArray,
    val numTargetFrames: Int,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TrainingSample) return false
        return numFrames == other.numFrames &&
            numTargetFrames == other.numTargetFrames &&
            videoFrames.contentEquals(other.videoFrames) &&
            targetFeatures.contentEquals(other.targetFeatures)
    }

    override fun hashCode(): Int = numFrames * 31 + numTargetFrames
}
