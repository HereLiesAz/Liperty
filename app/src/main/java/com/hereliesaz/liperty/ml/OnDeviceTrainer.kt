package com.hereliesaz.liperty.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Handles On-Device Personalization (LoRA) using LiteRT Training Signatures.
 * This allows the model to adapt to the specific user's lip movements.
 */
class OnDeviceTrainer(private val context: Context) {

    private var interpreter: Interpreter? = null
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

            val options = Interpreter.Options()
            // Training usually requires CPU as GPU delegates rarely support training ops
            
            interpreter = Interpreter(modelFileInFilesDir, options)
            Log.i("OnDeviceTrainer", "Trainable LoRA Model loaded from ${modelFileInFilesDir.absolutePath}")
        } catch (e: Exception) {
            Log.e("OnDeviceTrainer", "Failed to load trainable model", e)
        }
    }

    /**
     * Runs a single training step (batch) to update model weights.
     * @param inputBuffer ByteBuffer containing video frames [Batch, Time, Height, Width, Channels]
     * @param labelBuffer ByteBuffer containing one-hot encoded labels [Batch, Time, Classes]
     * @return The loss value for this step.
     */
    fun trainStep(inputBuffer: ByteBuffer, labelBuffer: ByteBuffer): Float {
        val interpreter = interpreter ?: return -1f

        // Inputs map matching the signature defined in create_trainable_model.py
        val inputs = mapOf(
            "video_input" to inputBuffer,
            "target_labels" to labelBuffer
        )

        // Outputs map
        val lossBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
        val outputs = mapOf(
            "loss" to lossBuffer
        )

        try {
            // Run the 'train' signature
            interpreter.runSignature(inputs, outputs, "train")
            
            lossBuffer.rewind()
            return lossBuffer.float
        } catch (e: Exception) {
            Log.e("OnDeviceTrainer", "Training step failed", e)
            return -1f
        }
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
        interpreter?.close()
    }
}
