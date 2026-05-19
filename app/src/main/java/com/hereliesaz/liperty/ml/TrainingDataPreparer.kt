package com.hereliesaz.liperty.ml

import com.hereliesaz.liperty.personalization.PairedTrainingRecord
import com.hereliesaz.liperty.personalization.PairedTrainingStore

/**
 * Converts [PairedTrainingRecord] instances into [TrainingSample] instances
 * suitable for the on-device LoRA trainer.
 *
 * For the MSE PoC the target is the frozen encoder's own output for
 * the same clip (so the adapter learns to approximate the identity
 * transform). This validates the pipeline end-to-end without needing
 * CTC labels. A follow-up replaces the target with CTC token indices
 * derived from the transcript field.
 */
class TrainingDataPreparer(
    private val encoder: EncoderSession,
    private val maxFrames: Int = 50,
) {
    /**
     * Prepare a single record into a training sample.
     * The record's [PairedTrainingRecord.videoFrames] are already
     * 88x88 normalized floats (mean=0.421, std=0.165 via
     * [VisemeCalibrationViewModel]).
     */
    fun prepare(record: PairedTrainingRecord): TrainingSample? {
        val frames = record.videoFrames
        if (frames.isEmpty()) return null

        val usedFrames = if (frames.size > maxFrames) {
            frames.takeLast(maxFrames)
        } else {
            frames
        }

        val T = usedFrames.size

        // Build NCTHW input: flatten (1, 1, T, 88, 88)
        val pixelsPerFrame = 88 * 88
        val videoFlat = FloatArray(T * pixelsPerFrame)
        for (i in usedFrames.indices) {
            System.arraycopy(usedFrames[i], 0, videoFlat, i * pixelsPerFrame, pixelsPerFrame)
        }

        // Run frozen encoder to get MSE target
        val inputShape = longArrayOf(1L, 1L, T.toLong(), 88L, 88L)
        val (features, featShape) = encoder.runEncode(videoFlat, inputShape)
        val tOut = featShape[1].toInt()

        return TrainingSample(
            videoFrames = videoFlat,
            numFrames = T,
            targetFeatures = features,
            numTargetFrames = tOut,
        )
    }

    /**
     * Prepare all records from a store. Filters out records with no
     * video frames.
     */
    fun prepareAll(store: PairedTrainingStore): List<TrainingSample> {
        return store.listAll().mapNotNull { id ->
            store.load(id)?.let { prepare(it) }
        }
    }
}
