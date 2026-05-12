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
 *   - **TODO: lip-ROI cropping** — the missing link. The raw bitmaps
 *     are full-frame faces; we need to run face landmarking and lip
 *     cropping over each to produce the 88×88 grayscale normalized
 *     [FloatArray]s the encoder expects.
 *   - [AudioTranscriber]: optional transcript labeling.
 *   - [PairedRecordExtractor] + [PairedTrainingStore]: assembly +
 *     persistence.
 *   - [PersonalizationConsentManager]: gates the whole flow on the
 *     user's explicit opt-in.
 *
 * Current implementation status: **partial**. The audio side, the
 * paired-record assembly, and the storage are working. The lip-ROI
 * cropping step is stubbed (returns empty frame lists, causing
 * [PairedRecordExtractor] to drop the segment). A complete
 * implementation requires sharing the live camera pipeline's
 * MediaPipe FaceLandmarker + `ImageUtils.alignAndCropMouth()` from
 * the offline import path. See the body comment of [processImport]
 * for the engineering required.
 *
 * Until the cropping integration lands, this class is wired in but
 * produces no [PairedTrainingRecord]s — the user can opt in and the
 * Settings UI shows zero records.
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

        val extractor = PairedRecordExtractor(
            extractVideoFrames = { _, startMs, durationMs ->
                // === REQUIRED FOLLOW-UP WORK ===
                // 1. videoFrameExtractor.extractFrames(uri, startMs, durationMs, fps=25)
                //    returns full-frame Bitmaps.
                // 2. For each Bitmap, run MediaPipe FaceLandmarker (shared with
                //    the live camera pipeline) to get lip landmarks.
                // 3. Apply ImageUtils.alignAndCropMouth() to crop the lip ROI
                //    to 88×88 grayscale, with the AV-HuBERT-matching
                //    mean=0.421/std=0.165 normalization.
                // 4. Flatten each crop to a FloatArray(88*88).
                //
                // The integration cost here is not the bitmap-to-FloatArray
                // conversion (cheap); it's wiring the FaceLandmarker so it
                // can be invoked from a background thread WITHOUT clobbering
                // the live-camera pipeline's instance. Either:
                //   (a) Spin up a separate FaceLandmarker instance for
                //       offline imports (clean, ~50 MB extra memory).
                //   (b) Pause the live pipeline, reuse its FaceLandmarker,
                //       resume — cheaper memory, brittler.
                //
                // Until this lands, return empty -> PairedRecordExtractor
                // drops the segment -> no records saved -> consent works
                // and stored count stays at 0.
                emptyList<FloatArray>()
            },
            transcriber = transcriber,
        )

        val records = extractor.extract(audioInputs, sourceUri = uri.toString())
        records.forEach { store.save(it) }
        Log.i(TAG, "processed $uri: ${records.size} records saved (of ${audioInputs.size} segments)")
        return records.size
    }

    companion object {
        private const val TAG = "PairedImportProcessor"
    }
}
