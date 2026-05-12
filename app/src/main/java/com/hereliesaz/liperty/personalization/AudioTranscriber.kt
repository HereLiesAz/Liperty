package com.hereliesaz.liperty.personalization

/**
 * Single ASR result returned by [AudioTranscriber.transcribe].
 *
 * @property text the recognized text
 * @property confidence ASR-reported confidence in [0, 1], or NaN if the
 *   backend doesn't expose one. Used by downstream training pipelines
 *   to filter noisy labels.
 */
data class TranscriptionResult(
    val text: String,
    val confidence: Float,
)

/**
 * Generates transcript labels for paired training recordings (see
 * [PairedTrainingRecord]). Implementations wrap an on-device ASR
 * backend; the audio is provided as a FloatArray of PCM samples to
 * avoid the file IO that Android's built-in [android.speech.SpeechRecognizer]
 * forces.
 *
 * Implementations must be **on-device only** — Liperty's privacy
 * contract forbids transmitting biometric data to remote services.
 */
interface AudioTranscriber {
    /**
     * Transcribe a single audio buffer.
     *
     * @param audioPcm mono PCM samples, normalized to [-1, 1]
     * @param sampleRate the buffer's sample rate (typically 16 kHz —
     *   matching [com.hereliesaz.liperty.voicebox.cloning.AudioPreprocessor]'s
     *   `TARGET_SAMPLE_RATE`)
     * @return the transcription or `null` when no recognition was
     *   possible (no backend installed, audio too short/noisy, etc.)
     */
    fun transcribe(audioPcm: FloatArray, sampleRate: Int): TranscriptionResult?

    fun close()
}

/**
 * Fallback no-op transcriber. Always returns `null`, never errors.
 *
 * Wired in when no real ASR backend is present so the rest of the
 * personalization pipeline can degrade gracefully — captured paired
 * recordings still produce valid [PairedTrainingRecord]s, just with
 * `transcript = null`. Downstream training layers that need text
 * (personal n-gram LM, encoder LoRA CTC supervision) filter these out;
 * unsupervised layers (viseme confusion statistics) use them as-is.
 *
 * **Real backend candidates for follow-up work:**
 * - [Vosk](https://alphacephei.com/vosk/android) — open-source on-device
 *   ASR with a small Android library. English model is ~50 MB. Strong
 *   accuracy on clean speech. Recommended first integration target.
 * - [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) — port of
 *   OpenAI Whisper to C++. tiny.en is 39 MB; base.en is 142 MB. Highest
 *   accuracy but bigger, slower, and requires NDK build work.
 * - Android's built-in [android.speech.SpeechRecognizer] is mic-only on
 *   most devices and not viable for transcribing an arbitrary audio
 *   buffer.
 *
 * The choice between Vosk and Whisper.cpp is largely a quality/size
 * tradeoff. For Liperty's use case (clean voice-clone audio of a single
 * known speaker), Vosk is likely sufficient and ships faster.
 */
class NoopAudioTranscriber : AudioTranscriber {
    override fun transcribe(audioPcm: FloatArray, sampleRate: Int): TranscriptionResult? = null
    override fun close() {}
}
