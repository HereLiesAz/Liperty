package com.hereliesaz.liperty.personalization

import java.util.UUID

/**
 * Orchestrator that combines speech-segmented audio, synchronized video
 * frames, and (optionally) ASR-derived transcripts into
 * [PairedTrainingRecord]s ready for [PairedTrainingStore].
 *
 * Designed for the voice-cloning import path: when the user imports a
 * video file, [com.hereliesaz.liperty.voicebox.cloning.AudioPreprocessor]
 * extracts speech segments via VAD. For each speech segment, this
 * extractor pulls the matching video frame window through the injected
 * [extractVideoFrames] lambda and runs the optional [transcriber].
 *
 * The video-extraction lambda is injected (rather than baking in
 * [VideoFrameExtractor]) so this orchestrator can be unit-tested
 * without Android `MediaMetadataRetriever` mocks — the live caller
 * binds `extractor::extractFrames` at construction time after the
 * `extractFrames` lambda is adapted to return lip-cropped 88×88
 * grayscale `FloatArray` frames rather than raw bitmaps.
 */
class PairedRecordExtractor(
    /** Lambda extracting a list of 88×88 grayscale normalized lip-ROI
     *  frames (as flat `FloatArray`s) for a given (sourceUri, startMs,
     *  durationMs) window. Empty list = no video track / extraction
     *  failed; the corresponding audio segment is dropped. */
    private val extractVideoFrames: (sourceUri: String, startMs: Long, durationMs: Long) -> List<FloatArray>,
    /** Transcript labeler. [NoopAudioTranscriber] when no ASR is wired. */
    private val transcriber: AudioTranscriber,
    /** PCM sample rate of `AudioInput.pcm`. Liperty's preprocessor
     *  resamples to 16 kHz, matching what most on-device ASRs expect. */
    private val sampleRate: Int = 16_000,
) {

    /** Adapter shape: the orchestrator only needs these fields from
     *  the audio side. Production callers map
     *  [com.hereliesaz.liperty.voicebox.cloning.AudioPreprocessor.SpeechSegment]
     *  to this; tests construct it directly. */
    data class AudioInput(
        val pcm: FloatArray,
        val startTimeMs: Long,
        val durationMs: Long,
    )

    /**
     * Process a list of audio segments + a source URI, returning the
     * paired training records ready for persistence. Segments whose
     * video extraction returns no frames are silently dropped.
     */
    fun extract(audioSegments: List<AudioInput>, sourceUri: String): List<PairedTrainingRecord> {
        val now = System.currentTimeMillis()
        return audioSegments.mapNotNull { seg ->
            val frames = extractVideoFrames(sourceUri, seg.startTimeMs, seg.durationMs)
            if (frames.isEmpty()) return@mapNotNull null
            val transcription = transcriber.transcribe(seg.pcm, sampleRate)
            PairedTrainingRecord(
                id = UUID.randomUUID().toString(),
                audioPcm = seg.pcm,
                videoFrames = frames,
                createdAtMs = now,
                source = sourceUri,
                transcript = transcription?.text,
                transcriptConfidence = transcription?.confidence,
            )
        }
    }
}
