package com.hereliesaz.liperty.ml

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Handles On-Device Personalization (LoRA) using TFLite Training Signatures.
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
     * Saves the updated model weights to internal storage so they persist across app restarts.
     * Note: TFLite Checkpoints are complex; often easier to save the specific weight buffers 
     * if the model architecture supports exporting them, or rely on OS file persistence if 
     * the interpreter was initialized from a writable file.
     * 
     * Since we loaded from Assets (read-only), we can't save *over* it.
     * In a real implementation, we would copy the asset to context.filesDir first,
     * load it from there, and then the updates are persisted in that file.
     */
    /**
     * Converts a list of phoneme strings into a ByteBuffer of float indices
     * using the canonical [MLConstants.PHONEME_VOCAB] ordering.
     * This replaces the previous dummy label approach where index mappings were arbitrary.
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
        // TFLite doesn't automatically "save" the file back to disk just by running inference/training.
        // The interpreter holds the state in RAM.
        // Exporting weights requires a specific "save" signature in the model or 
        // using the experimental variables export.
        Log.w("OnDeviceTrainer", "Model weight persistence requires 'save' signature implementation.")
    }

    fun close() {
        interpreter?.close()
    }
}
