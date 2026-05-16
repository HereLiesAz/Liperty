package com.hereliesaz.liperty.voicebox

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.io.Serializable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * PocketTTS Engine — OpenVoice v2 zero-shot voice cloning.
 *
 * Pipeline (two-stage TTS):
 * 1. **Base TTS** (`pocket_tts_base.onnx`): text token ids + base
 *    speaker id + speed → generic-voice waveform at 24 kHz.
 * 2. **Tone Color Converter** (`pocket_tts_tone_converter.onnx`):
 *    (base waveform, base speaker embedding, target speaker embedding)
 *    → user-voice waveform at 24 kHz.
 *
 * **Speaker Embedding Extractor** (`pocket_tts_se_extractor.onnx`):
 * (1, T) reference audio at 24 kHz → (1, 256, 1) speaker embedding.
 * Run on each user reference clip; the engine averages embeddings
 * across clips to converge toward the speaker's centroid in latent
 * space.
 *
 * **Tokenization** uses the symbol table in `pocket_tts_vocab.json`
 * (MeloTTS `g2p_en` phoneme set). Producing token ids from arbitrary
 * English text requires `g2p_en` to be ported to Kotlin (pending).
 * Until then, [generateAudio] returns null and [VoiceManager] falls
 * back to system TTS. Voice CLONING (embedding extraction + caching)
 * works end-to-end with just `pocket_tts_se_extractor.onnx`.
 *
 * **Why OpenVoice v2:** Liperty's user base includes people who lost
 * their voice (ALS, laryngectomy, stroke, vocal cord injury). They
 * have recordings of how they used to sound — voicemails, family
 * videos — and want to keep sounding like themselves. Zero-shot
 * cloning means a few seconds of reference audio is enough; no
 * fine-tuning required for first-class quality.
 */
class PocketTTSEngine(private val context: Context) {

    private val ortEnv = OrtEnvironment.getEnvironment()

    /** Stage 1: text + base speaker id → 24 kHz waveform (MeloTTS). */
    private var baseTtsSession: OrtSession? = null

    /** Reference audio → 256-d speaker embedding. */
    private var seExtractorSession: OrtSession? = null

    /** Stage 2: (audio, src_se, tgt_se) → converted 24 kHz waveform. */
    private var toneConverterSession: OrtSession? = null

    /**
     * Lazy-cached base speaker embedding (the "src_se" the tone
     * converter needs). Extracted on first synthesis by running the
     * SE extractor on a short base-TTS smoke utterance, then cached
     * across the process lifetime.
     */
    private var baseSpeakerEmbedding: FloatArray? = null

    /** Tokenizer for MeloTTS input_ids — see class kdoc. */
    private var tokenizerLoaded: Boolean = false

    /** True when both the base TTS and SE extractor are loaded. */
    val isReady: Boolean
        get() = baseTtsSession != null && seExtractorSession != null

    /** True when full text→audio synthesis is reachable. */
    val canSynthesize: Boolean
        get() = isReady && toneConverterSession != null && tokenizerLoaded

    companion object {
        private const val TAG = "PocketTTSEngine"

        // ── OpenVoice v2 asset names ───────────────────────────────
        private const val BASE_TTS_MODEL = "pocket_tts_base.onnx"
        private const val SE_EXTRACTOR_MODEL = "pocket_tts_se_extractor.onnx"
        private const val TONE_CONVERTER_MODEL = "pocket_tts_tone_converter.onnx"
        private const val VOCAB_FILE = "pocket_tts_vocab.json"

        /** Rejects 9-byte HTML stubs left by failed downloads. */
        private const val MIN_MODEL_SIZE_BYTES = 1_000L

        /**
         * OpenVoice v2 output sample rate. Downstream AudioTrack /
         * AudioRouter MUST use this rate.
         */
        const val TTS_OUTPUT_SAMPLE_RATE_HZ = 24_000

        /**
         * Sample rate the SE extractor and tone converter expect
         * for reference / source audio. Anything else is resampled
         * via [linearResample] before feeding the model.
         */
        const val SE_EXTRACTOR_SAMPLE_RATE_HZ = 24_000

        /** Speaker embedding dimensionality (OpenVoice v2). */
        const val EMBEDDING_DIM = 256

        /**
         * Default base speaker id used by the MeloTTS base model
         * when no specific voice is requested. The exported vocab
         * JSON lists `available_base_speakers` — index 0 is the
         * neutral "EN-Default".
         */
        const val DEFAULT_BASE_SPEAKER_ID: Long = 0L

        /**
         * Default synthesis speed multiplier (1.0 = normal).
         * Lower = slower, higher = faster.
         */
        const val DEFAULT_SPEED: Float = 1.0f

        /**
         * Tone-conversion temperature. OpenVoice v2 reference value;
         * higher = stronger timbre transfer but more artifacts.
         */
        const val DEFAULT_TAU: Float = 0.3f
    }

    /**
     * Loads ONNX sessions and the vocab table. Each section is
     * wrapped in try/catch so a missing optional model doesn't
     * block the others.
     */
    fun initialize() {
        // --- Vocab (required for synthesis; cloning works without it) ---
        try {
            val vocab = loadTextFile(VOCAB_FILE)
            tokenizerLoaded = vocab != null && vocab.isNotEmpty()
            if (tokenizerLoaded) {
                Log.i(TAG, "Vocab loaded (${vocab!!.length} bytes).")
            } else {
                Log.w(TAG, "Vocab not present — text→audio disabled, cloning still works.")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Vocab load failed: ${e.message}")
            tokenizerLoaded = false
        }

        // --- Base TTS (required for synthesis) ---
        baseTtsSession = openSession(BASE_TTS_MODEL)
        // --- SE extractor (required for cloning) ---
        seExtractorSession = openSession(SE_EXTRACTOR_MODEL)
        // --- Tone converter (required for synthesis in user's voice) ---
        toneConverterSession = openSession(TONE_CONVERTER_MODEL)

        Log.i(
            TAG,
            "PocketTTS init — base=${baseTtsSession != null} " +
                "se=${seExtractorSession != null} " +
                "conv=${toneConverterSession != null} " +
                "vocab=$tokenizerLoaded canSynth=$canSynthesize"
        )
    }

    private fun openSession(fileName: String): OrtSession? {
        return try {
            val file = deployModel(fileName)
            if (file != null) {
                ortEnv.createSession(file.absolutePath).also {
                    Log.i(TAG, "Loaded $fileName.")
                }
            } else null
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load $fileName: ${e.message}")
            null
        }
    }

    /**
     * Copies an asset to filesDir if absent, rejecting stubs.
     */
    private fun deployModel(fileName: String): File? {
        val file = File(context.filesDir, fileName)
        if (file.exists() && file.length() >= MIN_MODEL_SIZE_BYTES) return file
        if (file.exists()) file.delete()

        return try {
            context.assets.open(fileName).use { input ->
                file.outputStream().use { output -> input.copyTo(output) }
            }
            if (file.length() < MIN_MODEL_SIZE_BYTES) {
                Log.w(TAG, "$fileName too small (${file.length()} B) — likely a stub.")
                file.delete()
                null
            } else file
        } catch (e: Exception) {
            Log.w(TAG, "Asset $fileName not found: ${e.message}")
            if (file.exists()) file.delete()
            null
        }
    }

    private fun loadTextFile(fileName: String): String? {
        val file = File(context.filesDir, fileName)
        if (file.exists() && file.length() > 0) return file.readText()
        return try {
            context.assets.open(fileName).bufferedReader().use { it.readText() }
        } catch (_: Exception) { null }
    }

    // ── Voice Cloning (embedding extraction) ──────────────────────────

    /**
     * Clones a voice from a single reference WAV file.
     *
     * Returns a [VoiceState] with a 256-d embedding ready to be
     * persisted by [com.hereliesaz.liperty.voicebox.cloning.VoiceStore]
     * and consumed by [generateAudio].
     */
    fun cloneVoice(audioFile: File): VoiceState = cloneVoice(
        name = audioFile.nameWithoutExtension,
        audioFiles = listOf(audioFile)
    )

    /**
     * Clones a voice from N reference WAV files, averaging the
     * extracted 256-d embeddings.
     *
     * Multi-clip averaging is the on-device "more samples =
     * sounds more like you" lever: each clip's embedding is a
     * noisy estimate of the speaker's true centroid in latent
     * space; averaging converges toward that centroid as N grows.
     * Empirically this is most of the quality gain that doesn't
     * require fine-tuning.
     *
     * Quality bands (rough, OpenVoice v2):
     * - 1 clip,  ~5 s  → Initial (recognizable)
     * - 5 clips, ~2 min → Good
     * - 15 clips, ~10 min → Excellent (centroid-stable)
     * - 30+ min via Tier-3 fine-tune → Personalized
     */
    fun cloneVoice(name: String, audioFiles: List<File>): VoiceState {
        require(audioFiles.isNotEmpty()) { "Audio files list cannot be empty" }
        val embeddings = audioFiles.map { extractEmbedding(it) }
        val averaged = averageEmbeddings(embeddings)
        return VoiceState(name = name, embedding = averaged)
    }

    /**
     * Per-element mean of a non-empty list of equally-sized
     * embeddings. Throws if the dimensions disagree (which would
     * indicate a stale/legacy 192-d profile mixed with new 256-d
     * extractions — never silently average across dim mismatches).
     */
    internal fun averageEmbeddings(embeddings: List<FloatArray>): FloatArray {
        require(embeddings.isNotEmpty())
        val dim = embeddings[0].size
        require(embeddings.all { it.size == dim }) {
            "Embedding dim mismatch: ${embeddings.map { it.size }}"
        }
        val averaged = FloatArray(dim)
        for (emb in embeddings) {
            for (i in 0 until dim) averaged[i] += emb[i]
        }
        val n = embeddings.size.toFloat()
        for (i in 0 until dim) averaged[i] /= n
        return averaged
    }

    /**
     * Extracts a (256-d) speaker embedding from raw PCM mono audio
     * already resampled to [SE_EXTRACTOR_SAMPLE_RATE_HZ] (24 kHz)
     * and scaled to [-1, 1] float32.
     *
     * Stays in RAM only — no temporary files. The output is
     * flattened from the model's (1, 256, 1) shape to a 256-d array.
     */
    fun extractEmbeddingFromPcm(pcmSamples: FloatArray): FloatArray {
        val session = seExtractorSession
            ?: throw IllegalStateException("SE extractor not initialized")

        OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(pcmSamples),
            longArrayOf(1, pcmSamples.size.toLong())
        ).use { audioTensor ->
            val inputs = mapOf(seExtractorInputName(session) to audioTensor)
            session.run(inputs).use { result ->
                val tensor = result.get(0)?.value as? OnnxTensor
                    ?: throw IllegalStateException("SE extractor produced no output")
                val buf = tensor.floatBuffer
                val out = FloatArray(buf.remaining())
                buf.get(out)
                // Output is (1, 256, 1); flatten — already in (256,) order.
                return out
            }
        }
    }

    /** OpenVoice v2 SE extractor accepts `audio` as its input name. */
    private fun seExtractorInputName(session: OrtSession): String {
        val names = session.inputNames
        return when {
            names.contains("audio") -> "audio"
            names.contains("input") -> "input"
            names.contains("wav") -> "wav"
            else -> names.first()
        }
    }

    private fun extractEmbedding(audioFile: File): FloatArray {
        Log.i(TAG, "Extracting embedding from: ${audioFile.name}")
        val (samples, sampleRate) = readWavFile(audioFile)
        val resampled = if (sampleRate == SE_EXTRACTOR_SAMPLE_RATE_HZ) {
            samples
        } else {
            linearResample(samples, sampleRate, SE_EXTRACTOR_SAMPLE_RATE_HZ)
        }
        return extractEmbeddingFromPcm(resampled)
    }

    /**
     * Minimal WAV reader (16-bit PCM mono). Honors the sample rate
     * declared in the header so a user-supplied 16/22.05/44.1/48 kHz
     * clip (voicemail, video extract, voice memo) can be resampled
     * to the 24 kHz the extractor expects.
     *
     * Falls back to assuming 24 kHz mono PCM if the header is
     * unparseable, matching the previous behavior.
     */
    private fun readWavFile(file: File): Pair<FloatArray, Int> {
        val bytes = file.readBytes()
        if (bytes.size < 44 || String(bytes, 0, 4) != "RIFF") {
            // Not a WAV header — assume raw 16-bit PCM at 24 kHz.
            return decodePcm16(bytes, 0) to SE_EXTRACTOR_SAMPLE_RATE_HZ
        }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val sampleRate = bb.getInt(24)
        val numChannels = bb.getShort(22).toInt()
        val bitsPerSample = bb.getShort(34).toInt()

        // Find the "data" subchunk — its offset is not always 44.
        var dataOffset = 12
        while (dataOffset < bytes.size - 8) {
            val tag = String(bytes, dataOffset, 4)
            val size = bb.getInt(dataOffset + 4)
            if (tag == "data") {
                dataOffset += 8
                break
            }
            dataOffset += 8 + size
        }
        if (dataOffset >= bytes.size) {
            return decodePcm16(bytes, 44) to SE_EXTRACTOR_SAMPLE_RATE_HZ
        }

        if (bitsPerSample != 16) {
            Log.w(TAG, "WAV bitsPerSample=$bitsPerSample (only 16 supported); falling back.")
        }
        val mono = decodePcm16Channels(bytes, dataOffset, numChannels)
        return mono to sampleRate
    }

    private fun decodePcm16(bytes: ByteArray, headerOffset: Int): FloatArray {
        val n = maxOf(0, (bytes.size - headerOffset) / 2)
        val out = FloatArray(n)
        var idx = headerOffset
        for (i in 0 until n) {
            val sample = ((bytes[idx + 1].toInt() shl 8) or
                (bytes[idx].toInt() and 0xFF)).toShort()
            out[i] = sample.toFloat() / Short.MAX_VALUE
            idx += 2
        }
        return out
    }

    private fun decodePcm16Channels(bytes: ByteArray, offset: Int, channels: Int): FloatArray {
        if (channels <= 1) return decodePcm16(bytes, offset)
        val frameBytes = channels * 2
        val frames = (bytes.size - offset) / frameBytes
        val out = FloatArray(frames)
        for (i in 0 until frames) {
            var sum = 0
            val frameStart = offset + i * frameBytes
            for (c in 0 until channels) {
                val s = ((bytes[frameStart + c * 2 + 1].toInt() shl 8) or
                    (bytes[frameStart + c * 2].toInt() and 0xFF)).toShort()
                sum += s.toInt()
            }
            out[i] = (sum.toFloat() / channels) / Short.MAX_VALUE
        }
        return out
    }

    /**
     * Naive linear resampler. Good enough for voice-quality SE
     * extraction; for ultra-low-noise inputs we should swap in a
     * windowed-sinc filter later. Public for unit testing.
     */
    internal fun linearResample(input: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
        if (srcRate == dstRate || input.isEmpty()) return input
        val ratio = dstRate.toDouble() / srcRate.toDouble()
        val outLen = (input.size * ratio).toInt()
        val out = FloatArray(outLen)
        for (i in 0 until outLen) {
            val srcIdxD = i / ratio
            val i0 = srcIdxD.toInt()
            val i1 = (i0 + 1).coerceAtMost(input.size - 1)
            val frac = (srcIdxD - i0).toFloat()
            out[i] = input[i0] * (1f - frac) + input[i1] * frac
        }
        return out
    }

    // ── TTS Generation (two-stage: base → tone convert) ─────────────────

    /**
     * Generates audio from text incrementally (word-by-word streaming).
     * Each chunk is synthesized end-to-end through both stages.
     */
    fun generateAudioStreaming(text: String, voiceState: VoiceState): Sequence<FloatArray> {
        return sequence {
            val words = text.trim().split(Regex("\\s+"))
            for (word in words) {
                if (word.isNotBlank()) {
                    val chunk = generateAudio(word, voiceState)
                    if (chunk != null) yield(chunk)
                }
            }
        }
    }

    /**
     * Generates audio for [text] in [voiceState]'s timbre.
     *
     * Pipeline:
     * 1. Tokenize text → input_ids (currently unavailable — needs
     *    Kotlin g2p_en port; returns null until then).
     * 2. Run base TTS → generic-voice waveform at 24 kHz.
     * 3. Extract / cache base speaker embedding (src_se).
     * 4. Run tone converter with (waveform, src_se, tgt_se=voiceState.embedding).
     * 5. Return converted waveform at 24 kHz.
     *
     * Returns null on any failure so [VoiceManager] can fall back
     * to system TTS without surfacing the failure to the user.
     */
    fun generateAudio(text: String, voiceState: VoiceState): FloatArray? {
        val base = baseTtsSession ?: return null
        val converter = toneConverterSession
        val seExtractor = seExtractorSession

        if (voiceState.embedding.size != EMBEDDING_DIM) {
            Log.w(
                TAG,
                "Voice profile '${voiceState.name}' has embedding dim " +
                    "${voiceState.embedding.size}, expected $EMBEDDING_DIM. " +
                    "Profile is from a previous engine version — re-record to clone."
            )
            return null
        }

        val tokens = tokenize(text) ?: run {
            // Tokenizer not yet available on device.
            Log.d(TAG, "Tokenizer unavailable — synthesis skipped, falling back.")
            return null
        }
        if (tokens.isEmpty()) return null

        return try {
            // Stage 1: base TTS.
            val baseAudio = runBaseTts(base, tokens) ?: return null

            // No tone converter or SE extractor → fall back to base voice.
            if (converter == null || seExtractor == null) return baseAudio

            // src_se: cached base speaker embedding (extracted lazily).
            val srcSe = baseSpeakerEmbedding ?: run {
                val extracted = extractEmbeddingFromPcm(baseAudio)
                baseSpeakerEmbedding = extracted
                extracted
            }

            runToneConverter(converter, baseAudio, srcSe, voiceState.embedding)
        } catch (e: Exception) {
            Log.e(TAG, "TTS generation failed", e)
            null
        }
    }

    /**
     * Placeholder tokenizer. The MeloTTS pipeline needs
     * `g2p_en`-equivalent text→phoneme conversion ported to Kotlin,
     * keyed by the symbol table in `pocket_tts_vocab.json`.
     * Returns null until that lands; engine + VoiceManager will
     * fall back to system TTS in the meantime, preserving
     * end-to-end voice CLONING (which doesn't need this path).
     */
    private fun tokenize(@Suppress("UNUSED_PARAMETER") text: String): LongArray? {
        if (!tokenizerLoaded) return null
        // TODO(tier-2): port g2p_en to Kotlin and emit MeloTTS input_ids.
        return null
    }

    private fun runBaseTts(session: OrtSession, tokens: LongArray): FloatArray? {
        OnnxTensor.createTensor(
            ortEnv,
            LongBuffer.wrap(tokens),
            longArrayOf(1, tokens.size.toLong())
        ).use { tokenTensor ->
            OnnxTensor.createTensor(
                ortEnv,
                LongBuffer.wrap(longArrayOf(DEFAULT_BASE_SPEAKER_ID)),
                longArrayOf()
            ).use { sidTensor ->
                OnnxTensor.createTensor(
                    ortEnv,
                    FloatBuffer.wrap(floatArrayOf(DEFAULT_SPEED)),
                    longArrayOf()
                ).use { speedTensor ->
                    val inputs = mapOf(
                        "input_ids" to tokenTensor,
                        "speaker_id" to sidTensor,
                        "speed" to speedTensor
                    )
                    session.run(inputs).use { result ->
                        return extractFloatOutput(result)
                    }
                }
            }
        }
    }

    private fun runToneConverter(
        session: OrtSession,
        sourceAudio: FloatArray,
        srcSe: FloatArray,
        tgtSe: FloatArray
    ): FloatArray? {
        // Tone converter expects (1, 256, 1) embedding shape.
        val srcSeShaped = reshape256x1(srcSe)
        val tgtSeShaped = reshape256x1(tgtSe)

        OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(sourceAudio),
            longArrayOf(1, sourceAudio.size.toLong())
        ).use { audioTensor ->
            OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(srcSeShaped),
                longArrayOf(1, EMBEDDING_DIM.toLong(), 1)
            ).use { srcSeTensor ->
                OnnxTensor.createTensor(
                    ortEnv,
                    FloatBuffer.wrap(tgtSeShaped),
                    longArrayOf(1, EMBEDDING_DIM.toLong(), 1)
                ).use { tgtSeTensor ->
                    OnnxTensor.createTensor(
                        ortEnv,
                        FloatBuffer.wrap(floatArrayOf(DEFAULT_TAU)),
                        longArrayOf()
                    ).use { tauTensor ->
                        // OpenVoice v2 export uses these names; if the
                        // session exposes alternates, route through the
                        // first three audio-shaped inputs by position.
                        val inputs = buildConverterInputs(
                            session, audioTensor, srcSeTensor, tgtSeTensor, tauTensor
                        )
                        session.run(inputs).use { result ->
                            return extractFloatOutput(result)
                        }
                    }
                }
            }
        }
    }

    private fun buildConverterInputs(
        session: OrtSession,
        audio: OnnxTensor,
        srcSe: OnnxTensor,
        tgtSe: OnnxTensor,
        tau: OnnxTensor
    ): Map<String, OnnxTensor> {
        val expected = session.inputNames.toList()
        val map = mutableMapOf<String, OnnxTensor>()
        // Documented contract: source_audio, src_se, tgt_se, tau.
        if (expected.contains("source_audio")) map["source_audio"] = audio
        else if (expected.contains("audio_src")) map["audio_src"] = audio
        else if (expected.contains("audio")) map["audio"] = audio
        else map[expected[0]] = audio

        if (expected.contains("src_se")) map["src_se"] = srcSe
        else if (expected.size > 1) map[expected[1]] = srcSe

        if (expected.contains("tgt_se")) map["tgt_se"] = tgtSe
        else if (expected.size > 2) map[expected[2]] = tgtSe

        if (expected.contains("tau")) map["tau"] = tau

        return map
    }

    /** Ensures the embedding is exactly EMBEDDING_DIM floats; pads or trims if not. */
    private fun reshape256x1(emb: FloatArray): FloatArray {
        if (emb.size == EMBEDDING_DIM) return emb
        val out = FloatArray(EMBEDDING_DIM)
        emb.copyInto(out, 0, 0, minOf(emb.size, EMBEDDING_DIM))
        return out
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
        baseTtsSession?.close()
        seExtractorSession?.close()
        toneConverterSession?.close()
        baseTtsSession = null
        seExtractorSession = null
        toneConverterSession = null
        baseSpeakerEmbedding = null
        // OrtEnvironment is a singleton; other engines share it.
    }
}

/**
 * Lightweight voice identity for runtime inference. The embedding
 * is the OpenVoice v2 256-d speaker embedding (averaged across
 * however many reference clips the user provided).
 *
 * Persisted via [com.hereliesaz.liperty.voicebox.cloning.VoiceStore].
 * Profiles persisted by the previous (192-d ECAPA) engine will be
 * detected by dim mismatch in [PocketTTSEngine.generateAudio] and
 * skipped with a re-record prompt rather than silently misused.
 */
data class VoiceState(
    val name: String,
    val embedding: FloatArray
) : Serializable
