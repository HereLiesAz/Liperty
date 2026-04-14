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
    private var inputWidth = 128
    private var inputHeight = 64
    private var numFrames = 50
    private var numChannels = 3

    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: ByteBuffer? = null
    private var landmarkBuffer: ByteBuffer? = null

    /**
     * Initializes the inference engine.
     * Call this from a background thread.
     */
    fun initialize(): Boolean {
        return engine.initialize()
    }

    private fun allocateBuffersIfNeeded(outputShape: IntArray, landmarkShape: IntArray) {
        val inputCapacity = 1 * numFrames * inputHeight * inputWidth * numChannels * 4
        if (inputBuffer == null || inputBuffer!!.capacity() != inputCapacity) {
            inputBuffer = ByteBuffer.allocateDirect(inputCapacity).order(ByteOrder.nativeOrder())
        }

        if (outputShape.size >= 3) {
            val batchSize = outputShape[0]
            val timeSteps = outputShape[1]
            val vocabSize = outputShape[2]
            val outputCapacity = batchSize * timeSteps * vocabSize * 4
            if (outputBuffer == null || outputBuffer!!.capacity() != outputCapacity) {
                outputBuffer = ByteBuffer.allocateDirect(outputCapacity).order(ByteOrder.nativeOrder())
            }
        }

        if (landmarkShape.isNotEmpty()) {
            val landmarkElements = landmarkShape.reduce { acc, i -> acc * i }
            val landmarkCapacity = landmarkElements * 4
            if (landmarkBuffer == null || landmarkBuffer!!.capacity() != landmarkCapacity) {
                landmarkBuffer = ByteBuffer.allocateDirect(landmarkCapacity).order(ByteOrder.nativeOrder())
            }
        }
    }

    /**
     * Runs inference on the provided frames and (optionally) lip landmarks.
     */
    fun runInference(frames: List<Bitmap>, landmarks: FloatArray? = null): VSRResult {
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

            val outputShape = engine.getOutputShape(0)
            val landmarkShape = try { engine.getInputShape(1) } catch (e: Exception) { intArrayOf() }
            
            if (landmarks != null) {
                if (landmarkShape.size != 3 || landmarkShape[1] != 50 || landmarkShape[2] != 40) {
                    throw IllegalArgumentException("Landmarks provided but model second input shape ${landmarkShape.contentToString()} does not match expected [1, 50, 40].")
                }
            }

            Log.d("VSRInference", "outputShape=${outputShape.contentToString()}")
            if (outputShape.size < 3) {
                Log.e("VSRInference", "Unexpected output shape: ${outputShape.contentToString()}")
                val processingTime = SystemClock.uptimeMillis() - startTime
                return VSRResult("", 0f, emptyList(), processingTime)
            }

            allocateBuffersIfNeeded(outputShape, landmarkShape)

            val currentInputBuffer = inputBuffer!!
            val currentOutputBuffer = outputBuffer!!
            currentInputBuffer.rewind()
            currentOutputBuffer.rewind()

            // 1. Prepare Input Buffer
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
                        currentInputBuffer.putFloat(r / 255.0f)
                    } else if (numChannels == 3) {
                        // Extract RGB
                        val r = (pixel shr 16) and 0xFF
                        val g = (pixel shr 8) and 0xFF
                        val b = pixel and 0xFF
                        currentInputBuffer.putFloat(r / 255.0f)
                        currentInputBuffer.putFloat(g / 255.0f)
                        currentInputBuffer.putFloat(b / 255.0f)
                    }
                }
            }

            // Pad with zeros if fewer frames
            val paddingFrames = numFrames - framesToProcess.size
            for (i in 0 until paddingFrames * inputHeight * inputWidth * numChannels) {
                currentInputBuffer.putFloat(0f)
            }

            currentInputBuffer.rewind()

            // Diagnostic: log mean pixel value of frame 0 to confirm input varies across batches
            run {
                val slice = FloatArray(inputHeight * inputWidth * numChannels)
                currentInputBuffer.asFloatBuffer().get(slice)
                currentInputBuffer.rewind()
                val mean = slice.average()
                Log.d("VSRInput", "input buffer frame0 mean=%.4f (%.1f/255)".format(mean, mean * 255))
            }

            // 2. Prepare Output Buffer (already allocated)
            val batchSize = outputShape[0]
            val timeSteps = outputShape[1]
            val vocabSize = outputShape[2]
            Log.d("VSRInference", "decoding: timeSteps=$timeSteps vocabSize=$vocabSize")

            // 3. Run Inference
            if (landmarks != null && landmarkShape.isNotEmpty()) {
                // Secondary input buffer for LipCoordNet landmarks
                val currentLandmarkBuffer = landmarkBuffer!!
                currentLandmarkBuffer.rewind()

                val floatBuffer = currentLandmarkBuffer.asFloatBuffer()
                val landmarkElements = landmarkShape.reduce { acc, i -> acc * i }

                if (landmarks.size >= landmarkElements) {
                    floatBuffer.put(landmarks, 0, landmarkElements)
                } else {
                    floatBuffer.put(landmarks)
                    for (i in landmarks.size until landmarkElements) {
                        floatBuffer.put(0f)
                    }
                }

                engine.run(arrayOf(currentInputBuffer, currentLandmarkBuffer), currentOutputBuffer)
            } else {
                engine.run(currentInputBuffer, currentOutputBuffer)
            }

            currentOutputBuffer.rewind()

            // 4. Decode
            // CTC models output raw logits. Apply softmax per timestep so the
            // BeamSearchDecoder receives proper probabilities in [0,1].
            val probabilities = Array(timeSteps) {
                val logits = FloatArray(vocabSize) { currentOutputBuffer.float }
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
