package com.hereliesaz.liperty.ml

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Interpreter
import com.google.ai.edge.litert.gpu.CompatibilityList
import com.google.ai.edge.litert.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * VoiceConverter: Real-time neural mapping from Electrolarynx 'robotic' speech
 * to a natural 'target' human voice.
 *
 * Uses a GAN-based architecture (StarGAN-v2-VC or modified MelGAN) optimized for
 * on-device TFLite inference.
 */
class VoiceConverter(context: Context) {

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    init {
        val options = com.google.ai.edge.litert.Interpreter.Options()
        val compatList = com.google.ai.edge.litert.gpu.CompatibilityList()

        if (compatList.isDelegateSupportedOnThisDevice) {
            gpuDelegate = com.google.ai.edge.litert.gpu.GpuDelegate()
            options.addDelegate(gpuDelegate)
        } else {
            options.setNumThreads(4)
        }

        val modelBuffer = loadModelFile(context, "voice_converter.tflite")
        interpreter = com.google.ai.edge.litert.Interpreter(modelBuffer, options)
    }

    /**
     * Converts robotic speech Mel-spectrogram to human-like Mel-spectrogram.
     * @param inputMel Mel-spectrogram of the Electrolarynx speech.
     * @return Converted Mel-spectrogram of the target voice.
     */
    fun convert(inputMel: FloatArray, width: Int, height: Int): FloatArray {
        val inputBuffer = ByteBuffer.allocateDirect(width * height * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        inputBuffer.put(inputMel)

        val outputBuffer = ByteBuffer.allocateDirect(width * height * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        interpreter?.run(inputBuffer, outputBuffer)

        val result = FloatArray(width * height)
        outputBuffer.rewind()
        outputBuffer.get(result)
        return result
    }

    private fun loadModelFile(context: Context, modelName: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }
}
