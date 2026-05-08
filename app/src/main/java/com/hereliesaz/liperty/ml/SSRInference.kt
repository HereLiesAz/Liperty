package com.hereliesaz.liperty.ml

import android.os.SystemClock
import android.util.Log
import com.hereliesaz.liperty.utils.PerformanceMonitor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Handles inference for Silent Speech Recognition (SSR).
 * Translates bone vibrations / physiological signals into text.
 */
class SSRInference(private val engine: ModelEngine) {

    private val greedyDecoder = GreedyDecoder()
    private val beamDecoder = BeamSearchDecoder()
    private var useBeamSearch = true

    private var cachedInputBuffer: ByteBuffer? = null
    private var cachedOutputBuffer: ByteBuffer? = null

    fun initialize(): Boolean {
        return engine.initialize()
    }

    /**
     * Runs SSR inference on raw vibration signal.
     * @param signal Raw PCM float signal from BCM/Accelerometer.
     */
    fun runInference(signal: FloatArray): VSRResult {
        val startTime = SystemClock.uptimeMillis()

        try {
            val inputShape = engine.getInputShape(0)
            if (inputShape.isEmpty()) {
                 val processingTime = SystemClock.uptimeMillis() - startTime
                 return VSRResult("", 0f, emptyList(), processingTime)
            }

            // Typically: [Batch, Time, Features]
            val numSteps = inputShape[1]
            val numFeatures = inputShape[2]

            val inputCapacity = 1 * numSteps * numFeatures * 4
            if (cachedInputBuffer == null || cachedInputBuffer!!.capacity() != inputCapacity) {
                cachedInputBuffer = ByteBuffer.allocateDirect(inputCapacity).order(ByteOrder.nativeOrder())
            }
            val inputBuffer = cachedInputBuffer!!
            inputBuffer.rewind()

            // Reshape/Window the signal to fit the model input if necessary.
            // For now, we assume the signal matches the expected feature count.
            for (i in 0 until (numSteps * numFeatures).coerceAtMost(signal.size)) {
                inputBuffer.putFloat(signal[i])
            }
            // Padding
            val remaining = (numSteps * numFeatures) - signal.size
            if (remaining > 0) {
                for (i in 0 until remaining) inputBuffer.putFloat(0f)
            }
            inputBuffer.rewind()

            val outputShape = engine.getOutputShape(0)
            if (outputShape.size < 3) {
                 Log.e("SSRInference", "Unexpected output shape")
                 val processingTime = SystemClock.uptimeMillis() - startTime
                 return VSRResult("", 0f, emptyList(), processingTime)
            }

            val batchSize = outputShape[0]
            val timeSteps = outputShape[1]
            val vocabSize = outputShape[2]

            val outputCapacity = batchSize * timeSteps * vocabSize * 4
            if (cachedOutputBuffer == null || cachedOutputBuffer!!.capacity() != outputCapacity) {
                cachedOutputBuffer = ByteBuffer.allocateDirect(outputCapacity).order(ByteOrder.nativeOrder())
            }
            val outputBuffer = cachedOutputBuffer!!
            outputBuffer.rewind()

            engine.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()

            val probabilities = Array(timeSteps) {
                val probs = FloatArray(vocabSize)
                for (v in 0 until vocabSize) {
                    probs[v] = outputBuffer.float
                }
                probs
            }

            val decodedText = if (useBeamSearch) {
                beamDecoder.decode(probabilities)
            } else {
                greedyDecoder.decode(probabilities)
            }

            val confidence = if (probabilities.isNotEmpty()) {
                probabilities.map { it.maxOrNull() ?: 0f }.average().toFloat()
            } else 0f

            val processingTime = SystemClock.uptimeMillis() - startTime
            return VSRResult(decodedText, confidence, emptyList(), processingTime)

        } catch (e: Exception) {
            Log.e("SSRInference", "SSR Inference Failed", e)
            val processingTime = SystemClock.uptimeMillis() - startTime
            return VSRResult("", 0f, emptyList(), processingTime)
        }
    }

    fun close() {
        engine.close()
    }
}
