package com.hereliesaz.liperty.personalization

import android.content.Context
import android.net.Uri
import android.util.Log
import com.hereliesaz.liperty.voicebox.cloning.AudioPreprocessor

/**
 * Top-level orchestrator for harvesting paired (audio, lip-video,
 * transcript) training data from a voice-cloning import flow.
 *
 * Glues together:
 *   - [AudioPreprocessor]: extracts speech segments from the imported
 *     audio/video URI via VAD.
 *   - [VideoFrameExtractor]: pulls raw bitmaps from the same URI at
 *     each speech segment's time window.
 *   - **lip-ROI cropping**: a per-import [com.hereliesaz.liperty.ml.FaceLandmarkerHelper]
 *     instance runs face landmarking over each extracted bitmap and
 *     [LipCropPipeline.cropMouthRoi] produces the 88×88 grayscale
 *     normalized [FloatArray]s the encoder expects — reproducing the
 *     live camera preprocessing exactly so training data matches the
 *     inference distribution.
 *   - [AudioTranscriber]: optional transcript labeling.
 *   - [PairedRecordExtractor] + [PairedTrainingStore]: assembly +
 *     persistence.
 *   - [PersonalizationConsentManager]: gates the whole flow on the
 *     user's explicit opt-in.
 *
 * Current implementation status: **functional**. The audio side, the
 * lip-ROI cropping (via [LipCropPipeline]), the paired-record
 * assembly, and the storage are all wired end-to-end. The remaining
 * personalization gaps are downstream of this class: the consumer of
 * the saved [PairedTrainingRecord]s (Step 2 statistical personalization
 * and Step 3 on-device encoder LoRA training) — see `docs/PERSONALIZATION.md`.
 */
class PairedImportProcessor(
    private val context: Context,
    private val audioPreprocessor: AudioPreprocessor,
    private val videoFrameExtractor: VideoFrameExtractor,
    private val store: PairedTrainingStore,
    private val consent: PersonalizationConsentManager,
    private val transcriber: AudioTranscriber = NoopAudioTranscriber(),
) {

    /**
     * Process a single URI through the personalization pipeline.
     * Returns the number of [PairedTrainingRecord]s saved, or 0 if
     * consent is not granted or the URI has no video / no speech.
     */
    fun processImport(uri: Uri): Int {
        if (!consent.hasConsent()) {
            Log.i(TAG, "skipping import — personalization consent not granted")
            return 0
        }
        if (!videoFrameExtractor.hasVideoTrack(uri)) {
            Log.i(TAG, "skipping import — URI has no video track")
            return 0
        }

        // Audio: VAD + segmentation via the existing voice-cloning preprocessor.
        val pre = try {
            audioPreprocessor.processUri(uri, uri.lastPathSegment ?: "import")
        } catch (e: Exception) {
            Log.w(TAG, "audio preprocess failed: ${e.message}")
            return 0
        }
        if (pre.segments.isEmpty()) {
            Log.i(TAG, "no speech segments in $uri")
            return 0
        }

        val audioInputs = pre.segments.map { seg ->
            PairedRecordExtractor.AudioInput(
                pcm = seg.pcmData,
                startTimeMs = seg.startTimeMs,
                durationMs = seg.durationMs,
            )
        }

        // Offline lip-ROI cropping uses a SEPARATE FaceLandmarker instance
        // (option (a)) so it never clobbers the live camera pipeline's
        // @Synchronized instance. ~50 MB while an import runs; closed in the
        // finally below. The crop reproduces the live preprocessing exactly via
        // [LipCropPipeline] so training data matches the inference distribution.
        val importLandmarker = com.hereliesaz.liperty.ml.FaceLandmarkerHelper(context)
        try {
            val extractor = PairedRecordExtractor(
                extractVideoFrames = { _, startMs, durationMs ->
                    videoFrameExtractor.extractFrames(uri, startMs, durationMs, fps = TARGET_FPS)
                        .mapNotNull { bmp ->
                            try {
                                val result = importLandmarker.detectSynchronously(bmp)
                                    ?: return@mapNotNull null
                                LipCropPipeline.cropMouthRoi(
                                    bitmap = bmp,
                                    result = result,
                                    helper = importLandmarker,
                                    cropSize = CROP_SIZE,
                                    pixelMean = PIXEL_MEAN,
                                    pixelStd = PIXEL_STD,
                                    mirror = false,
                                )
                            } finally {
                                // Source frames are full-resolution; recycle each
                                // immediately so long imports don't OOM.
                                bmp.recycle()
                            }
                        }
                },
                transcriber = transcriber,
            )

            val records = extractor.extract(audioInputs, sourceUri = uri.toString())
            records.forEach { store.save(it) }
            Log.i(TAG, "processed $uri: ${records.size} records saved (of ${audioInputs.size} segments)")
            return records.size
        } finally {
            importLandmarker.close()
        }
    }

    companion object {
        private const val TAG = "PairedImportProcessor"

        // MUST match MainActivity.AUTOAVSR_* / VSRInference so offline training
        // data is preprocessed identically to live inference. See LipCropPipeline.
        private const val CROP_SIZE = 88
        private const val PIXEL_MEAN = 0.421f
        private const val PIXEL_STD = 0.165f
        private const val TARGET_FPS = 25
    }
}
