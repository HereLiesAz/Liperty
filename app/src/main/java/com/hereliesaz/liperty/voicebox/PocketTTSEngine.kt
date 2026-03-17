package com.hereliesaz.liperty.voicebox

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.io.Serializable
import java.nio.FloatBuffer
import com.hereliesaz.liperty.ml.G2PConverter
import com.hereliesaz.liperty.ml.MLConstants

/**
 * PocketTTS Engine for on-device voice cloning and TTS.
 * Uses ONNX Runtime to execute exported Pocket-TTS models.
 */
class PocketTTSEngine(private val context: Context) {

    private val ortEnv = OrtEnvironment.getEnvironment()
    private var acousticSession: OrtSession? = null
    private var vocoderSession: OrtSession? = null
    private var speakerEncoderSession: OrtSession? = null
    private var voiceConversionSession: OrtSession? = null
    private val converter = G2PConverter()

    companion object {
        private const val TAG = "PocketTTSEngine"
        private const val ACOUSTIC_MODEL = "pocket_tts_acoustic.onnx"
        private const val VOCODER_MODEL = "pocket_tts_vocoder.onnx"
        private const val SPEAKER_ENCODER_MODEL = "pocket_tts_speaker.onnx"
        private const val VC_MODEL = "pocket_tts_vc.onnx"
    }

    /**
     * Initializes the engine by loading the ONNX models.
     */
    fun initialize() {
        try {
            // Deploy models from assets if not already in filesDir
            val acousticModelFile = copyFromAssetsIfMissing(ACOUSTIC_MODEL)
            val vocoderModelFile = copyFromAssetsIfMissing(VOCODER_MODEL)
            val speakerModelFile = copyFromAssetsIfMissing(SPEAKER_ENCODER_MODEL)
            val vcModelFile = copyFromAssetsIfMissing(VC_MODEL)

            if (acousticModelFile == null || vocoderModelFile == null) {
                Log.e(TAG, "Critical PocketTTS models (acoustic/vocoder) could not be deployed.")
                return
            }

            acousticSession = ortEnv.createSession(acousticModelFile.absolutePath)
            vocoderSession = ortEnv.createSession(vocoderModelFile.absolutePath)

            if (speakerModelFile != null) speakerEncoderSession = ortEnv.createSession(speakerModelFile.absolutePath)
            if (vcModelFile != null) voiceConversionSession = ortEnv.createSession(vcModelFile.absolutePath)

            Log.i(TAG, "PocketTTS Engine initialized successfully.")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing PocketTTS Engine: ${e.message}")
        }
    }

    private fun copyFromAssetsIfMissing(fileName: String): File? {
        val file = File(context.filesDir, fileName)
        if (file.exists()) return file

        return try {
            context.assets.open(fileName).use { inputStream ->
                file.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            Log.i(TAG, "Copied $fileName from assets to internal storage.")
            file
        } catch (e: Exception) {
            Log.w(TAG, "Asset $fileName not found or could not be copied: ${e.message}")
            if (file.exists()) file.delete()
            null
        }
    }

    /**
     * Performs voice conversion mapping source audio to a target voice profile.
     */
    fun performVoiceConversion(sourceAudio: FloatArray, targetVoice: VoiceState): FloatArray? {
        val vcSession = voiceConversionSession ?: return null
        try {
            return OnnxTensor.createTensor(ortEnv, java.nio.FloatBuffer.wrap(sourceAudio), longArrayOf(1, sourceAudio.size.toLong())).use { audioTensor ->
                OnnxTensor.createTensor(ortEnv, java.nio.FloatBuffer.wrap(targetVoice.embedding), longArrayOf(1, targetVoice.embedding.size.toLong())).use { voiceTensor ->
                    val inputs = mapOf("source_audio" to audioTensor, "target_embedding" to voiceTensor)
                    vcSession.run(inputs).use { result ->
                        val tensor = result.get(0)?.value as? OnnxTensor
                        if (tensor != null) {
                            val floatBuffer = tensor.floatBuffer
                            val audioOutput = FloatArray(floatBuffer.remaining())
                            floatBuffer.get(audioOutput)
                            audioOutput
                        } else {
                            null
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Voice conversion failed", e)
            return null
        }
    }

    /**
     * Extracts a voice state from a reference audio file using the speaker encoder model.
     */
    fun cloneVoice(audioFile: File): VoiceState {
        val session = speakerEncoderSession ?: throw IllegalStateException("Speaker encoder not initialized")
        Log.i(TAG, "Cloning voice from: ${audioFile.name}")
        
        // Minimal PCM load logic: skip 44-byte WAV header
        val bytes = audioFile.readBytes()
        val headerOffset = 44
        val numSamples = Math.max(0, (bytes.size - headerOffset) / 2)
        val floatSamples = FloatArray(numSamples)
        for (i in 0 until numSamples) {
            val byteIndex = headerOffset + (i * 2)
            val sample = ((bytes[byteIndex + 1].toInt() shl 8) or (bytes[byteIndex].toInt() and 0xFF)).toShort()
            floatSamples[i] = sample.toFloat() / Short.MAX_VALUE
        }

        var embedding: FloatArray? = null
        OnnxTensor.createTensor(ortEnv, java.nio.FloatBuffer.wrap(floatSamples), longArrayOf(1, floatSamples.size.toLong())).use { audioTensor ->
            val inputs = mapOf("audio" to audioTensor)
            session.run(inputs).use { result ->
                val tensor = result.get(0)?.value as? OnnxTensor ?: throw IllegalStateException("Failed to extract embedding")
                val floatBuffer = tensor.floatBuffer
                embedding = FloatArray(floatBuffer.remaining())
                floatBuffer.get(embedding)
            }
        }

        return VoiceState(
            name = audioFile.nameWithoutExtension,
            embedding = embedding ?: FloatArray(0)
        )
    }

    /**
     * Generates audio from text incrementally as words/tokens become available.
     * Provides streaming output for ultra-low latency TTS.
     */
    fun generateAudioStreaming(text: String, voiceState: VoiceState, vocoderInputName: String = "mel"): Sequence<FloatArray> {
        return sequence {
            // Split text into meaningful chunks (e.g., sentences or phrases)
            // For SSR, words come in one by one, but VALL-E models benefit from context.
            // Here we prioritize latency: synthesize word-by-word if needed, or buffer slightly.
            val tokens = text.trim().split(Regex("\\s+"))
            for (token in tokens) {
                if (token.isNotBlank()) {
                    val audioChunk = generateAudio(token, voiceState, vocoderInputName)
                    if (audioChunk != null) {
                        yield(audioChunk)
                    }
                }
            }
        }
    }

    /**
     * Generates audio from text using a specific voice state.
     */
    fun generateAudio(text: String, voiceState: VoiceState, vocoderInputName: String = "mel"): FloatArray? {
        val session = acousticSession ?: return null
        
        try {
            // 1. Tokenize (Using pre-initialized field)
            val phonemes = converter.sentenceToPhonemes(text)
            
            // Map phonemes to indices according to MLConstants.PHONEME_VOCAB
            // Since tokens are longs in ONNX, we convert to LongArray
            val vocab = MLConstants.PHONEME_VOCAB
            val tokens = phonemes.map { phoneme ->
                val idx = vocab.indexOf(phoneme)
                if (idx >= 0) idx.toLong() else 0L // 0 usually for blank/unknown in VALLR
            }.toLongArray()
            
            // Handle edge case where no phonemes could be generated
            if (tokens.isEmpty()) return null
            
            return OnnxTensor.createTensor(ortEnv, java.nio.LongBuffer.wrap(tokens), longArrayOf(1, tokens.size.toLong())).use { tokenTensor ->
                OnnxTensor.createTensor(ortEnv, java.nio.FloatBuffer.wrap(voiceState.embedding), longArrayOf(1, voiceState.embedding.size.toLong())).use { voiceTensor ->
                    // 2. Acoustic Model: (tokens, voice) -> mel-spectrogram
                    val inputs = mapOf("input_ids" to tokenTensor, "speaker_ids" to voiceTensor)
                    session.run(inputs).use { result ->
                        // 3. Vocoder: (mel-spectrogram) -> PCM
                        val vocoder = vocoderSession ?: return@use null
                        val melTensorOpt = result.get(0)?.value as? OnnxTensor ?: return@use null

                        val vocoderInputs = mapOf(vocoderInputName to melTensorOpt)
                        vocoder.run(vocoderInputs).use { vocoderResult ->
                            Log.i(TAG, "Generated audio for text: $text")
                            val tensor = vocoderResult.get(0)?.value as? OnnxTensor
                            if (tensor != null) {
                                val floatBuffer = tensor.floatBuffer
                                val audioOutput = FloatArray(floatBuffer.remaining())
                                floatBuffer.get(audioOutput)
                                audioOutput
                            } else null
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "TTS Generation failed", e)
            return null
        }
    }

    fun close() {
        acousticSession?.close()
        vocoderSession?.close()
        speakerEncoderSession?.close()
        voiceConversionSession?.close()
        ortEnv.close()
    }
}

/**
 * Represents the identity of a cloned voice.
 */
data class VoiceState(
    val name: String,
    val embedding: FloatArray // The latent vector representing the voice
) : Serializable
