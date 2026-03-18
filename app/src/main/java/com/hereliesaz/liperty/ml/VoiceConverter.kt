package com.hereliesaz.liperty.ml

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel

/**
 * VoiceConverter: Real-time neural mapping from Electrolarynx 'robotic' speech
 * to a natural 'target' human voice.
 *
 * Uses a GAN-based architecture (StarGAN-v2-VC or modified MelGAN) optimized for
 * on-device LiteRT inference.
 */
class VoiceConverter(context: Context) {

    private var compiledModel: CompiledModel? = null

    init {
        compiledModel = try {
            CompiledModel.create(context.assets, "voice_converter.tflite", CompiledModel.Options(Accelerator.GPU))
        } catch (e: Exception) {
            Log.w("VoiceConverter", "GPU compilation failed, falling back to CPU", e)
            try {
                CompiledModel.create(context.assets, "voice_converter.tflite", CompiledModel.Options(Accelerator.CPU))
            } catch (e2: Exception) {
                Log.e("VoiceConverter", "Failed to load voice_converter.tflite", e2)
                null
            }
        }
    }

    /**
     * Converts robotic speech Mel-spectrogram to human-like Mel-spectrogram.
     * @param inputMel Mel-spectrogram of the Electrolarynx speech.
     * @return Converted Mel-spectrogram of the target voice.
     */
    fun convert(inputMel: FloatArray, width: Int, height: Int): FloatArray {
        val model = compiledModel ?: return FloatArray(width * height)

        val inputBuffers = model.createInputBuffers()
        val outputBuffers = model.createOutputBuffers()

        inputBuffers[0].writeFloat(inputMel)
        model.run(inputBuffers, outputBuffers)

        return outputBuffers[0].readFloat()
    }

    fun close() {
        compiledModel?.close()
    }
}
