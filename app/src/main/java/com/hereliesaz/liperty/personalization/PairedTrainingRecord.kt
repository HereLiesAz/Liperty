package com.hereliesaz.liperty.personalization

/**
 * One paired (audio, video, optional-transcript) sample of THIS user
 * speaking, captured during voice cloning so we can also use it for
 * personalizing the visual ASR pipeline (Step 1 of the on-device
 * personalization work — see docs/AVHUBERT_V3_BACKEND.md "tenth attempt").
 *
 * Important constraints:
 *   - Video is stored as 88×88 grayscale crops (Liperty's encoder input
 *     format), NOT raw camera frames. The lip-ROI crop happens during
 *     capture; we never persist faces or other identifying features.
 *   - Audio is 16 kHz mono float32 PCM (same as voice cloning expects).
 *   - The transcript is optional. When present (filled by Android
 *     SpeechRecognizer on the recorded audio), it's the supervision
 *     signal for downstream LoRA training. When absent, this record is
 *     still useful for unsupervised statistics (viseme confusions, lip
 *     motion baselines).
 *   - These records are LEGAL-CLASS biometric data under BIPA/GDPR.
 *     They MUST live only in the dedicated training directory gated by
 *     a separate consent flow, never in regular app storage. Deletion
 *     via [PairedTrainingStore.deleteAll] must wipe them physically,
 *     not just unlink.
 */
data class PairedTrainingRecord(
    /** Unique id; doubles as the filesystem stem (`<id>.audio`, `<id>.video`, `<id>.json`). */
    val id: String,
    /** 16 kHz mono float32 PCM samples. */
    val audioPcm: FloatArray,
    /** Lip-cropped 88×88 grayscale frames (NCHW one frame at a time as
     *  `FloatArray` of size 88*88, AV-HuBERT mean/std-normalized).
     *  Indexed by frame, then by pixel row-major. */
    val videoFrames: List<FloatArray>,
    /** Wall-clock millis at capture time (for retention policy). */
    val createdAtMs: Long,
    /** Source URI / file name the import came from, for the user to
     *  cross-reference when reviewing/deleting stored recordings. */
    val source: String,
    /** Optional ASR-derived transcript. Null when we couldn't transcribe
     *  (no SpeechRecognizer available, audio too noisy, etc.). */
    val transcript: String? = null,
    /** ASR confidence score in [0, 1], if the recognizer provided one.
     *  Used by training pipelines to weight samples and filter noisy labels. */
    val transcriptConfidence: Float? = null,
) {
    /** Sample count assertion — audio is expected to be `durationSeconds * 16000`. */
    val durationMs: Long get() = (audioPcm.size * 1000L) / 16_000L

    override fun equals(other: Any?): Boolean {
        // dataclass-default equals on FloatArray compares by reference; we
        // override so tests can compare records by content where it matters.
        if (this === other) return true
        if (other !is PairedTrainingRecord) return false
        if (id != other.id) return false
        if (createdAtMs != other.createdAtMs) return false
        if (source != other.source) return false
        if (transcript != other.transcript) return false
        if (!audioPcm.contentEquals(other.audioPcm)) return false
        if (videoFrames.size != other.videoFrames.size) return false
        for (i in videoFrames.indices) {
            if (!videoFrames[i].contentEquals(other.videoFrames[i])) return false
        }
        return true
    }

    override fun hashCode(): Int = id.hashCode() * 31 + createdAtMs.hashCode()
}
