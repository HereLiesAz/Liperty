package com.hereliesaz.liperty.ml

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.hereliesaz.liperty.utils.PerformanceMonitor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

data class VSRResult(
    val text: String,
    val confidence: Float,
    val wordConfidences: List<Float> = emptyList(),
    val processingTimeMs: Long
)

class VSRInference(private val engine: ModelEngine) {

    private val greedyDecoder = GreedyDecoder()
    private val beamDecoder = BeamSearchDecoder()
    private var useBeamSearch = true

    // Default Constants (overridden dynamically by model shape)
    private var inputWidth = 88
    private var inputHeight = 88
    private var numFrames = 50
    private var numChannels = 1

    /**
     * Initializes the inference engine.
     * Call this from a background thread.
     */
    fun initialize(): Boolean {
        return engine.initialize()
    }

    /**
     * Runs inference on the provided frames.
     */
    fun runInference(frames: List<Bitmap>): VSRResult {
        val startTime = SystemClock.uptimeMillis()

        try {
            val inputShape = engine.getInputShape(0)
            Log.d("VSRInference", "inputShape=${inputShape.contentToString()} frames=${frames.size}")
            if (inputShape.isNotEmpty()) {
                // Determine model type based on shape
                if (inputShape.size >= 5) {
                    numFrames = inputShape[1]
                    inputHeight = inputShape[2]
                    inputWidth = inputShape[3]
                    numChannels = inputShape[4]
                }
            }
            Log.d("VSRInference", "using numFrames=$numFrames ${inputWidth}x${inputHeight} ch=$numChannels")
            
            // 1. Prepare Input Buffer
            // Float32 (4 bytes)
            val inputBuffer = ByteBuffer.allocateDirect(1 * numFrames * inputHeight * inputWidth * numChannels * 4)
            inputBuffer.order(ByteOrder.nativeOrder())

            // Take last N frames if we have more, or all if fewer
            val framesToProcess = if (frames.size > numFrames) {
                frames.takeLast(numFrames)
            } else {
                frames
            }

            for (bitmap in framesToProcess) {
                // Resize if needed
                val scaledBitmap = if (bitmap.width != inputWidth || bitmap.height != inputHeight) {
                    Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
                } else {
                    bitmap
                }

                val pixels = IntArray(inputWidth * inputHeight)
                scaledBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

                for (pixel in pixels) {
                    if (numChannels == 1) {
                        // Extract Red channel for grayscale
                        val r = (pixel shr 16) and 0xFF
                        inputBuffer.putFloat(r / 255.0f)
                    } else if (numChannels == 3) {
                        // Extract RGB
                        val r = (pixel shr 16) and 0xFF
                        val g = (pixel shr 8) and 0xFF
                        val b = pixel and 0xFF
                        inputBuffer.putFloat(r / 255.0f)
                        inputBuffer.putFloat(g / 255.0f)
                        inputBuffer.putFloat(b / 255.0f)
                    }
                }
            }

            // Pad with zeros if fewer frames
            val paddingFrames = numFrames - framesToProcess.size
            for (i in 0 until paddingFrames * inputHeight * inputWidth * numChannels) {
                inputBuffer.putFloat(0f)
            }

            inputBuffer.rewind()

            // Diagnostic: log mean pixel value of frame 0 to confirm input varies across batches
            run {
                val slice = FloatArray(inputHeight * inputWidth * numChannels)
                inputBuffer.asFloatBuffer().get(slice)
                inputBuffer.rewind()
                val mean = slice.average()
                Log.d("VSRInput", "input buffer frame0 mean=%.4f (%.1f/255)".format(mean, mean * 255))
            }

            // 2. Prepare Output Buffer
            val outputShape = engine.getOutputShape(0)

            Log.d("VSRInference", "outputShape=${outputShape.contentToString()}")
            if (outputShape.size < 3) {
                Log.e("VSRInference", "Unexpected output shape: ${outputShape.contentToString()}")
                val processingTime = SystemClock.uptimeMillis() - startTime
                return VSRResult("", 0f, emptyList(), processingTime)
            }

            val batchSize = outputShape[0]
            val timeSteps = outputShape[1]
            val vocabSize = outputShape[2]
            Log.d("VSRInference", "decoding: timeSteps=$timeSteps vocabSize=$vocabSize")

            val outputBuffer = ByteBuffer.allocateDirect(batchSize * timeSteps * vocabSize * 4)
            outputBuffer.order(ByteOrder.nativeOrder())

            // 3. Run Inference
            engine.run(inputBuffer, outputBuffer)

            outputBuffer.rewind()

            // 4. Decode
            // CTC models output raw logits. Apply softmax per timestep so the
            // BeamSearchDecoder receives proper probabilities in [0,1].
            val probabilities = Array(timeSteps) {
                val logits = FloatArray(vocabSize) { outputBuffer.float }
                if (it == 0) {
                    Log.d("VSRInference", "logits[0] sample (first 5): ${logits.take(5)}")
                }
                softmax(logits)
            }

            val decodedText = if (useBeamSearch) {
                beamDecoder.decode(probabilities)
            } else {
                greedyDecoder.decode(probabilities)
            }

            // Compute confidence as mean max-softmax probability across all timesteps.
            // This reflects how "peaked" the model's output distribution was — a well-trained
            // model that recognised a clear phoneme will produce high max-prob values.
            val confidence = if (probabilities.isNotEmpty()) {
                probabilities.map { timeStep -> timeStep.max() }.average().toFloat()
            } else 0f

            // Assign the same confidence to every word produced by this inference window.
            val wordCount = decodedText.trim().split("\\s+".toRegex()).size.coerceAtLeast(1)
            val wordConfidences = List(wordCount) { confidence }

            val processingTime = SystemClock.uptimeMillis() - startTime
            PerformanceMonitor.logInferenceTime(processingTime)

            Log.d("VSRInference", "decoded='$decodedText' conf=%.2f time=${processingTime}ms".format(confidence))
            return VSRResult(decodedText, confidence, wordConfidences, processingTime)

        } catch (e: Exception) {
            Log.e("VSRInference", "Inference Failed", e)
            val processingTime = SystemClock.uptimeMillis() - startTime
            return VSRResult("", 0f, emptyList(), processingTime)
        }
    }

    fun close() {
        engine.close()
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }
}
