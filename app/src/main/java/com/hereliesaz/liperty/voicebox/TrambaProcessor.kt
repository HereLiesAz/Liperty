package com.hereliesaz.liperty.voicebox

import android.content.Context
import android.util.Log
import com.hereliesaz.liperty.ml.TFLiteEngine
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Implements the TRAMBA architecture for Audio Bandwidth Expansion.
 * Reconstructs high-frequency vocal components attenuated by tissue transmission
 * in Bone Conduction Microphones (BCMs).
 */
class TrambaProcessor(context: Context) {

    private val engine = TFLiteEngine(context, "tramba_model.tflite")
    private var isInitialized = false

    fun initialize(): Boolean {
        isInitialized = engine.initialize()
        if (isInitialized) {
            Log.i("TrambaProcessor", "TRAMBA Engine initialized successfully")
        } else {
            Log.e("TrambaProcessor", "Failed to initialize TRAMBA Engine")
        }
        return isInitialized
    }

    /**
     * Processes a buffer of PCM16 audio samples and returns the expanded (super-resolved) version.
     * Note: TRAMBA typically expects a specific window size (e.g., 512 or 1024 samples).
     */
    fun processAudio(inputPcm: ShortArray): ShortArray {
        if (!isInitialized) return inputPcm

        try {
            // TFLite models usually work with Float32
            val inputBuffer = ByteBuffer.allocateDirect(inputPcm.size * 4)
            inputBuffer.order(ByteOrder.nativeOrder())
            for (sample in inputPcm) {
                inputBuffer.putFloat(sample / 32768.0f)
            }
            inputBuffer.rewind()

            val outputBuffer = ByteBuffer.allocateDirect(inputPcm.size * 4)
            outputBuffer.order(ByteOrder.nativeOrder())

            engine.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()

            val outputPcm = ShortArray(inputPcm.size)
            for (i in outputPcm.indices) {
                val floatSample = outputBuffer.float
                outputPcm[i] = (floatSample * 32768.0f).toInt().coerceIn(-32768, 32767).toShort()
            }

            return outputPcm
        } catch (e: Exception) {
            Log.e("TrambaProcessor", "Error during TRAMBA inference", e)
            return inputPcm
        }
    }

    fun release() {
        engine.close()
    }
}
