package com.hereliesaz.liperty.ml

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.hereliesaz.liperty.utils.PerformanceMonitor
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class VSRResult(
    val text: String,
    val confidence: Float,
    val processingTimeMs: Long
)

class VSRInference(private val engine: ModelEngine) {

    private val greedyDecoder = GreedyDecoder()
    private val beamDecoder = BeamSearchDecoder()
    private var useBeamSearch = true

    // Constants (Assuming typical VSR model)
    private val INPUT_WIDTH = 88
    private val INPUT_HEIGHT = 88
    private val NUM_FRAMES = 50

    /**
     * Initializes the inference engine.
     * Call this from a background thread.
     */
    fun initialize() {
        engine.initialize()
    }

    /**
     * Runs inference on the provided frames.
     */
    fun runInference(frames: List<Bitmap>): VSRResult {
        val startTime = SystemClock.uptimeMillis()

        try {
            // 1. Prepare Input Buffer
            // Shape: [1, T, H, W, C] -> [1, 50, 88, 88, 1]
            // Float32 (4 bytes)
            val inputBuffer = ByteBuffer.allocateDirect(1 * NUM_FRAMES * INPUT_HEIGHT * INPUT_WIDTH * 1 * 4)
            inputBuffer.order(ByteOrder.nativeOrder())

            // Take last N frames if we have more, or all if fewer
            val framesToProcess = if (frames.size > NUM_FRAMES) {
                frames.takeLast(NUM_FRAMES)
            } else {
                frames
            }

            for (bitmap in framesToProcess) {
                // Resize if needed
                val scaledBitmap = if (bitmap.width != INPUT_WIDTH || bitmap.height != INPUT_HEIGHT) {
                    Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, true)
                } else {
                    bitmap
                }

                // Extract Grayscale values
                val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
                scaledBitmap.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)

                for (pixel in pixels) {
                    // Extract Red channel (since grayscale)
                    val r = (pixel shr 16) and 0xFF
                    // Normalize to 0-1
                    val normalized = r / 255.0f
                    inputBuffer.putFloat(normalized)
                }
            }

            // Pad with zeros if fewer frames
            val paddingFrames = NUM_FRAMES - framesToProcess.size
            for (i in 0 until paddingFrames * INPUT_HEIGHT * INPUT_WIDTH) {
                inputBuffer.putFloat(0f)
            }

            inputBuffer.rewind()

            // 2. Prepare Output Buffer
            val outputShape = engine.getOutputShape(0)

            if (outputShape.size < 3) {
                 Log.e("VSRInference", "Unexpected output shape: ${outputShape.contentToString()}")
                 return getDummyResult(frames.size, startTime)
            }

            val batchSize = outputShape[0]
            val timeSteps = outputShape[1]
            val vocabSize = outputShape[2]

            val outputBuffer = ByteBuffer.allocateDirect(batchSize * timeSteps * vocabSize * 4)
            outputBuffer.order(ByteOrder.nativeOrder())

            // 3. Run Inference
            engine.run(inputBuffer, outputBuffer)

            outputBuffer.rewind()

            // 4. Decode
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
            val processingTime = SystemClock.uptimeMillis() - startTime
            PerformanceMonitor.logInferenceTime(processingTime)

            return VSRResult(decodedText, 0.9f, processingTime)

        } catch (e: Exception) {
            Log.e("VSRInference", "Inference Failed", e)
            return getDummyResult(frames.size, startTime)
        }
    }

    private fun getDummyResult(frameCount: Int, startTime: Long): VSRResult {
        val processingTime = SystemClock.uptimeMillis() - startTime
        // In a real failure, we might return the last known good result or a blank
        return VSRResult("...", 0.0f, processingTime)
    }

    fun close() {
        engine.close()
    }
}
