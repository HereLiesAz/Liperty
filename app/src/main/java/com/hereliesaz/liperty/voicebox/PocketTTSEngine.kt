package com.hereliesaz.liperty.voicebox

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.io.Serializable

/**
 * PocketTTS Engine for on-device voice cloning and TTS.
 *
 * Uses ONNX Runtime to execute four models:
 * - **Speaker encoder** (SpeechBrain ECAPA-TDNN): audio → 192-d embedding
 * - **Acoustic model** (Coqui VITS VCTK): phoneme ids + speaker embedding → waveform
 * - **Vocoder** (pass-through identity): preserved for pipeline contract
 * - **Voice conversion** (optional FreeVC): source audio + target embedding → converted audio
 *
 * Tokenization uses [VitsTokenizer], which loads the model's own vocabulary
 * from `pocket_tts_phoneme_map.json` (exported alongside the ONNX by
 * `tools/export_tts_to_onnx.ipynb`). English text is converted to IPA
 * via [com.hereliesaz.liperty.ml.G2PConverter] and then indexed against
 * the VITS character vocabulary, with blank-token interleaving for
 * monotonic alignment.
 */
class PocketTTSEngine(private val context: Context) {

    private val ortEnv = OrtEnvironment.getEnvironment()
    private var acousticSession: OrtSession? = null
    private var vocoderSession: OrtSession? = null
    private var speakerEncoderSession: OrtSession? = null
    private var voiceConversionSession: OrtSession? = null

    private val tokenizer = VitsTokenizer()

    /** Whether the acoustic model output is a waveform (skip vocoder). */
    private var acousticOutputIsWaveform = false

    /** True when the acoustic session and tokenizer are both ready. */
    val isReady: Boolean
        get() = acousticSession != null && tokenizer.isLoaded

    companion object {
        private const val TAG = "PocketTTSEngine"
        private const val ACOUSTIC_MODEL = "pocket_tts_acoustic.onnx"
        private const val VOCODER_MODEL = "pocket_tts_vocoder.onnx"
        private const val SPEAKER_ENCODER_MODEL = "pocket_tts_speaker.onnx"
        private const val VC_MODEL = "pocket_tts_vc.onnx"
        private const val PHONEME_MAP = "pocket_tts_phoneme_map.json"

        /** Minimum file size to accept (rejects 9-byte HTML stubs). */
        private const val MIN_MODEL_SIZE_BYTES = 1_000L

        /**
         * Output sample rate of the bundled Coqui VITS VCTK acoustic
         * model. Downstream AudioTrack / AudioRouter MUST use this rate.
         */
        const val TTS_OUTPUT_SAMPLE_RATE_HZ = 22050

        /**
         * Input sample rate the speaker encoder (SpeechBrain ECAPA) expects.
         * Reference audio must be resampled to this rate.
         */
        const val SPEAKER_ENCODER_SAMPLE_RATE_HZ = 16000
    }

    /**
     * Initializes the engine: deploys assets, loads ONNX sessions,
     * and parses the phoneme map for tokenization.
     *
     * Each session is loaded independently so a failure in one
     * (e.g., missing optional VC model) doesn't block the others.
     */
    fun initialize() {
        // --- Phoneme map (required for TTS generation) ---
        try {
            val json = context.assets.open(PHONEME_MAP).bufferedReader().use { it.readText() }
            if (tokenizer.loadFromJson(json)) {
                Log.i(TAG, "Phoneme map loaded successfully.")
            } else {
                Log.w(TAG, "Phoneme map parsed but contained no char_to_id entries.")
            }
        } catch (e: Exception) {
            Log.w(TAG, "pocket_tts_phoneme_map.json not found — TTS generation disabled: ${e.message}")
        }

        // --- Acoustic model (required for TTS) ---
        try {
            val file = deployModel(ACOUSTIC_MODEL)
            if (file != null) {
                acousticSession = ortEnv.createSession(file.absolutePath)
                // Check if output is "waveform" (VITS end-to-end) vs "mel"
                acousticSession?.let { session ->
                    acousticOutputIsWaveform = session.outputNames.contains("waveform")
                }
                Log.i(TAG, "Acoustic model loaded (waveform output: $acousticOutputIsWaveform).")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load acoustic model: ${e.message}")
        }

        // --- Vocoder (optional — skipped when acoustic outputs waveform) ---
        try {
            val file = deployModel(VOCODER_MODEL)
            if (file != null) {
                vocoderSession = ortEnv.createSession(file.absolutePath)
                Log.i(TAG, "Vocoder loaded.")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Vocoder not loaded: ${e.message}")
        }

        // --- Speaker encoder (required for voice cloning) ---
        try {
            val file = deployModel(SPEAKER_ENCODER_MODEL)
            if (file != null) {
                speakerEncoderSession = ortEnv.createSession(file.absolutePath)
                Log.i(TAG, "Speaker encoder loaded.")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Speaker encoder not loaded: ${e.message}")
        }

        // --- Voice conversion (optional) ---
        try {
            val file = deployModel(VC_MODEL)
            if (file != null) {
                voiceConversionSession = ortEnv.createSession(file.absolutePath)
                Log.i(TAG, "Voice conversion model loaded.")
            }
        } catch (e: Exception) {
            Log.d(TAG, "Voice conversion model not available (optional): ${e.message}")
        }

        Log.i(TAG, "PocketTTS init complete. isReady=$isReady")
    }

    /**
     * Copies an asset to filesDir if not already present, validating
     * that the file is a real model (not a 9-byte stub).
     */
    private fun deployModel(fileName: String): File? {
        val file = File(context.filesDir, fileName)

        // If already deployed and valid size, reuse
        if (file.exists() && file.length() >= MIN_MODEL_SIZE_BYTES) return file

        // Delete any stale stub
        if (file.exists()) file.delete()

        return try {
            context.assets.open(fileName).use { input ->
                file.outputStream().use { output -> input.copyTo(output) }
            }
            if (file.length() < MIN_MODEL_SIZE_BYTES) {
                Log.w(TAG, "$fileName is too small (${file.length()} bytes) — likely a stub. Skipping.")
                file.delete()
                null
            } else {
                Log.i(TAG, "Deployed $fileName (${file.length()} bytes).")
                file
            }
        } catch (e: Exception) {
            Log.w(TAG, "Asset $fileName not found: ${e.message}")
            if (file.exists()) file.delete()
            null
        }
    }

    // ── Voice Conversion ─────────────────────────────────────────────────

    /**
     * Maps source audio to a target voice profile via the VC model.
     */
    fun performVoiceConversion(sourceAudio: FloatArray, targetVoice: VoiceState): FloatArray? {
        val vcSession = voiceConversionSession ?: return null
        return try {
            OnnxTensor.createTensor(ortEnv, java.nio.FloatBuffer.wrap(sourceAudio), longArrayOf(1, sourceAudio.size.toLong())).use { audioTensor ->
                OnnxTensor.createTensor(ortEnv, java.nio.FloatBuffer.wrap(targetVoice.embedding), longArrayOf(1, targetVoice.embedding.size.toLong())).use { voiceTensor ->
                    val inputs = mapOf("source_audio" to audioTensor, "target_embedding" to voiceTensor)
                    vcSession.run(inputs).use { result ->
                        extractFloatOutput(result)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Voice conversion failed", e)
            null
        }
    }

    // ── Voice Cloning (embedding extraction) ─────────────────────────────

    /**
     * Clones a voice from a single reference audio file.
     */
    fun cloneVoice(audioFile: File): VoiceState {
        return VoiceState(
            name = audioFile.nameWithoutExtension,
            embedding = extractEmbedding(audioFile)
        )
    }

    /**
     * Clones a voice from multiple reference files, averaging embeddings.
     */
    fun cloneVoice(name: String, audioFiles: List<File>): VoiceState {
        require(audioFiles.isNotEmpty()) { "Audio files list cannot be empty" }

        val embeddings = audioFiles.map { extractEmbedding(it) }
        val size = embeddings[0].size
        val averaged = FloatArray(size)
        for (i in 0 until size) {
            var sum = 0f
            for (emb in embeddings) sum += emb[i]
            averaged[i] = sum / embeddings.size
        }
        return VoiceState(name = name, embedding = averaged)
    }

    /**
     * Extracts a speaker embedding from raw PCM (16 kHz mono, [-1, 1]).
     * Audio stays in RAM only — no temporary files.
     */
    fun extractEmbeddingFromPcm(pcmSamples: FloatArray): FloatArray {
        val session = speakerEncoderSession
            ?: throw IllegalStateException("Speaker encoder not initialized")

        OnnxTensor.createTensor(
            ortEnv,
            java.nio.FloatBuffer.wrap(pcmSamples),
            longArrayOf(1, pcmSamples.size.toLong())
        ).use { audioTensor ->
            val inputs = mapOf("audio" to audioTensor)
            session.run(inputs).use { result ->
                val tensor = result.get(0)?.value as? OnnxTensor
                    ?: throw IllegalStateException("Failed to extract embedding")
                val buf = tensor.floatBuffer
                val embedding = FloatArray(buf.remaining())
                buf.get(embedding)
                return embedding
            }
        }
    }

    private fun extractEmbedding(audioFile: File): FloatArray {
        Log.i(TAG, "Extracting embedding from: ${audioFile.name}")
        val bytes = audioFile.readBytes()
        val headerOffset = 44
        val numSamples = maxOf(0, (bytes.size - headerOffset) / 2)
        val floatSamples = FloatArray(numSamples)
        for (i in 0 until numSamples) {
            val byteIndex = headerOffset + (i * 2)
            val sample = ((bytes[byteIndex + 1].toInt() shl 8) or (bytes[byteIndex].toInt() and 0xFF)).toShort()
            floatSamples[i] = sample.toFloat() / Short.MAX_VALUE
        }
        return extractEmbeddingFromPcm(floatSamples)
    }

    // ── TTS Generation ───────────────────────────────────────────────────

    /**
     * Generates audio from text incrementally (word-by-word streaming).
     */
    fun generateAudioStreaming(text: String, voiceState: VoiceState, vocoderInputName: String = "mel"): Sequence<FloatArray> {
        return sequence {
            val words = text.trim().split(Regex("\\s+"))
            for (word in words) {
                if (word.isNotBlank()) {
                    val chunk = generateAudio(word, voiceState, vocoderInputName)
                    if (chunk != null) yield(chunk)
                }
            }
        }
    }

    /**
     * Generates audio from text using a specific voice state.
     *
     * Tokenization flow:
     * 1. Text → IPA (via [G2PConverter] ARPABET→IPA) → character indices
     *    (via [VitsTokenizer] using the exported `char_to_id` map)
     * 2. Indices → ONNX acoustic model → waveform (or mel-spectrogram)
     * 3. If mel: vocoder → waveform. If waveform: return directly.
     *
     * Returns null if the engine is not ready (models/phoneme map missing)
     * or if inference fails. VoiceManager falls back to system TTS.
     */
    fun generateAudio(text: String, voiceState: VoiceState, vocoderInputName: String = "mel"): FloatArray? {
        val session = acousticSession ?: return null

        if (!tokenizer.isLoaded) {
            Log.w(TAG, "TTS disabled — phoneme map not loaded.")
            return null
        }

        val tokens = tokenizer.tokenize(text)
        if (tokens.isEmpty()) return null

        return try {
            OnnxTensor.createTensor(ortEnv, java.nio.LongBuffer.wrap(tokens), longArrayOf(1, tokens.size.toLong())).use { tokenTensor ->
                OnnxTensor.createTensor(ortEnv, java.nio.FloatBuffer.wrap(voiceState.embedding), longArrayOf(1, voiceState.embedding.size.toLong())).use { voiceTensor ->
                    val inputs = mapOf("input_ids" to tokenTensor, "speaker_ids" to voiceTensor)
                    session.run(inputs).use { result ->
                        if (acousticOutputIsWaveform) {
                            // VITS end-to-end: acoustic output IS waveform
                            extractFloatOutput(result)
                        } else {
                            // Two-stage: acoustic → mel, vocoder → waveform
                            val vocoder = vocoderSession ?: return@use null
                            val melTensor = result.get(0)?.value as? OnnxTensor ?: return@use null
                            vocoder.run(mapOf(vocoderInputName to melTensor)).use { vocResult ->
                                extractFloatOutput(vocResult)
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "TTS generation failed", e)
            null
        }
    }

    private fun extractFloatOutput(result: OrtSession.Result): FloatArray? {
        val tensor = result.get(0)?.value as? OnnxTensor ?: return null
        val buf = tensor.floatBuffer
        val output = FloatArray(buf.remaining())
        buf.get(output)
        return output
    }

    // ── Lifecycle ────────────────────────────────────────────────────────

    fun close() {
        acousticSession?.close()
        vocoderSession?.close()
        speakerEncoderSession?.close()
        voiceConversionSession?.close()
        // OrtEnvironment is a singleton; don't close it here as other
        // engines (OnnxModelEngine) may share it.
    }
}

/**
 * Lightweight voice identity for runtime inference.
 * Persisted via [com.hereliesaz.liperty.voicebox.cloning.VoiceStore].
 */
data class VoiceState(
    val name: String,
    val embedding: FloatArray
) : Serializable
