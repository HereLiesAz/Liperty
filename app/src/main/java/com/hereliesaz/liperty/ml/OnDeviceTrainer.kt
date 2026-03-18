package com.hereliesaz.liperty.ml

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Handles On-Device Personalization (LoRA) using LiteRT Training Signatures.
 * This allows the model to adapt to the specific user's lip movements.
 *
 * NOTE: LiteRT 2.x CompiledModel does not yet expose runSignature() for training.
 * The model is loaded for weight persistence and inference; the training step is
 * a no-op until the LiteRT training API stabilises.
 */
class OnDeviceTrainer(private val context: Context) {

    private var compiledModel: CompiledModel? = null
    private val MODEL_NAME = "vsr_lora_model.tflite"
    private val modelFileInFilesDir: File by lazy { File(context.filesDir, MODEL_NAME) }

    fun initialize() {
        try {
            // Copy model from assets to filesDir if not present, so we can update it
            if (!modelFileInFilesDir.exists()) {
                context.assets.open(MODEL_NAME).use { input ->
                    modelFileInFilesDir.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                Log.i("OnDeviceTrainer", "Copied base model to internal storage for personalization.")
            }

            // Training graphs require CPU; GPU delegates don't support training ops.
            compiledModel = try {
                CompiledModel.create(modelFileInFilesDir.absolutePath, CompiledModel.Options(Accelerator.CPU))
            } catch (e: Exception) {
                Log.w("OnDeviceTrainer", "CPU model load failed: ${e.message}")
                null
            }

            if (compiledModel != null) {
                Log.i("OnDeviceTrainer", "Trainable LoRA model loaded from ${modelFileInFilesDir.absolutePath}")
            }
        } catch (e: Exception) {
            Log.e("OnDeviceTrainer", "Failed to load trainable model", e)
        }
    }

    /**
     * Runs a single training step (batch) to update model weights.
     *
     * TODO: LiteRT 2.x CompiledModel does not expose runSignature("train").
     * Restore when the LiteRT training API is available.
     *
     * @return -1 until training is re-enabled.
     */
    fun trainStep(inputBuffer: ByteBuffer, labelBuffer: ByteBuffer): Float {
        Log.w("OnDeviceTrainer", "trainStep: on-device training requires LiteRT training API (not yet available in 2.x)")
        return -1f
    }

    /**
     * Converts a list of phoneme strings into a ByteBuffer of float indices
     * using the canonical [MLConstants.PHONEME_VOCAB] ordering.
     *
     * @param phonemes List of phoneme tokens (e.g. ["HH", "AH", "L", "OW"])
     * @return ByteBuffer ready to pass as `target_labels` to [trainStep]
     */
    fun createLabelBuffer(phonemes: List<String>): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(phonemes.size * 4).order(ByteOrder.nativeOrder())
        for (phoneme in phonemes) {
            val index = MLConstants.PHONEME_VOCAB.indexOf(phoneme)
            // Unknown phonemes map to blank (index 0) rather than -1 to stay within vocab bounds
            buf.putFloat(if (index >= 0) index.toFloat() else 0f)
        }
        buf.rewind()
        return buf
    }

    fun saveModelCheckpoint() {
        Log.w("OnDeviceTrainer", "Model weight persistence requires 'save' signature implementation.")
    }

    fun close() {
        compiledModel?.close()
    }
}
