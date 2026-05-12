package com.hereliesaz.liperty.ml

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.hereliesaz.liperty.utils.PerformanceMonitor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import com.hereliesaz.liperty.BuildConfig
import kotlin.math.exp

data class VSRResult(
    val text: String,
    val confidence: Float,
    val wordConfidences: List<Float> = emptyList(),
    val processingTimeMs: Long
)

/**
 * @property pixelMean Subtracted from `pixel/255` before division by [pixelStd].
 *   Defaults to 0 (i.e. plain `pixel/255`) for the legacy VideoMAE backend.
 *   Auto-AVSR uses 0.421.
 * @property pixelStd Divisor applied to `pixel/255 - pixelMean`. Defaults to 1.
 *   Auto-AVSR uses 0.165.
 * @property customDecoder If non-null, replaces the default ARPABET phoneme
 *   greedy/beam decoders. Receives `(T_out, V)` softmax probabilities.
 */
class VSRInference(
    private val engine: ModelEngine,
    private val pixelMean: Float = 0f,
    private val pixelStd: Float = 1f,
    private val customDecoder: ((Array<FloatArray>) -> String)? = null,
) {

    private val greedyDecoder = GreedyDecoder()
    private val beamDecoder = BeamSearchDecoder()
    private var useBeamSearch = true

    // Default Constants (overridden dynamically by model shape)
    private var inputWidth = 224
    private var inputHeight = 224
    private var numFrames = 16
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
    @Synchronized
    fun runInference(frames: List<Bitmap>, landmarks: FloatArray? = null): VSRResult {
        val startTime = SystemClock.uptimeMillis()

        try {
            val inputShape = engine.getInputShape(0)
            val inputLayout = engine.getInputLayout()
            Log.d("VSRInference", "inputShape=${inputShape.contentToString()} layout=$inputLayout frames=${frames.size}")
            parseInputShape(inputShape, inputLayout)?.let { dims ->
                numFrames = dims.numFrames
                inputHeight = dims.inputHeight
                inputWidth = dims.inputWidth
                numChannels = dims.numChannels
            }
            // ONNX models often leave the batch (N) and time (T) axes as -1
            // (dynamic). The buffer allocator below can't handle negatives;
            // substitute concrete values: T = the actual frame count passed
            // in by the caller, others fall back to the channel/spatial dims
            // the Auto-AVSR Conformer was trained on (1×88×88 grayscale).
            if (numFrames <= 0) numFrames = frames.size
            if (numChannels <= 0) numChannels = 1
            if (inputHeight <= 0) inputHeight = 88
            if (inputWidth <= 0) inputWidth = 88
            Log.d("VSRInference", "using numFrames=$numFrames ${inputWidth}x${inputHeight} ch=$numChannels layout=$inputLayout")

            val outputShape = engine.getOutputShape(0)
            val landmarkShape = try { engine.getInputShape(1) } catch (e: Exception) { intArrayOf() }
            
            if (landmarks != null && landmarkShape.isNotEmpty()) {
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

            // The output shape from ONNX has -1 for batch (N) and time (T_out).
            // CTC T_out depends on the encoder's stride; for Auto-AVSR it
            // typically equals numFrames (1:1 ratio per frame in the
            // visual-only Conformer). Use that as an upper bound so the
            // output buffer is sized correctly. Actual T_out from the
            // returned tensor is read after inference.
            val concreteOutputShape = IntArray(outputShape.size).also {
                outputShape.copyInto(it)
                if (it[0] <= 0) it[0] = 1                 // batch
                if (it[1] <= 0) it[1] = numFrames         // T_out upper bound
                // it[2] (vocab) is always concrete
            }
            allocateBuffersIfNeeded(concreteOutputShape, landmarkShape)

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

            // Extract the pixel data once per frame so the NCTHW path doesn't
            // need to re-decode bitmaps on each channel pass.
            val pixelsPerFrame = framesToProcess.map { bitmap ->
                val scaled = if (bitmap.width != inputWidth || bitmap.height != inputHeight) {
                    Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
                } else {
                    bitmap
                }
                val px = IntArray(inputWidth * inputHeight)
                scaled.getPixels(px, 0, inputWidth, 0, 0, inputWidth, inputHeight)
                if (scaled !== bitmap) scaled.recycle()
                px
            }
            val paddingFrames = numFrames - pixelsPerFrame.size
            val pixelsPerImage = inputHeight * inputWidth

            // Per-channel normalization: the model wants `(pixel/255 - mean) / std`.
            // For the legacy VideoMAE backend mean=0 std=1 → just pixel/255.
            // For Auto-AVSR mean=0.421 std=0.165.
            fun norm(raw255: Int): Float = (raw255 / 255f - pixelMean) / pixelStd

            when (inputLayout) {
                InputLayout.NTHWC -> {
                    for (px in pixelsPerFrame) {
                        for (pixel in px) {
                            when (numChannels) {
                                1 -> currentInputBuffer.putFloat(norm((pixel shr 16) and 0xFF))
                                3 -> {
                                    currentInputBuffer.putFloat(norm((pixel shr 16) and 0xFF))
                                    currentInputBuffer.putFloat(norm((pixel shr 8) and 0xFF))
                                    currentInputBuffer.putFloat(norm(pixel and 0xFF))
                                }
                            }
                        }
                    }
                    val padFloats = paddingFrames * pixelsPerImage * numChannels
                    repeat(padFloats) { currentInputBuffer.putFloat(0f) }
                }
                InputLayout.NCTHW -> {
                    // Layout: for each channel C, write all T frames' pixels.
                    // For ARGB packed ints: shift R=16, G=8, B=0 → shift = (2 - c) * 8
                    // when numChannels == 3. For grayscale (numChannels == 1) we
                    // convert to Rec.601 luma instead of just the red channel --
                    // Auto-AVSR was trained on proper grayscale conversion
                    // (cv2.cvtColor BGR2GRAY in Chaplin's preprocessing), so
                    // feeding a single colour channel is a distribution shift
                    // and the model collapses to predicting <blank>.
                    for (c in 0 until numChannels) {
                        val shift = if (numChannels == 1) -1 else (2 - c) * 8
                        for (px in pixelsPerFrame) {
                            for (pixel in px) {
                                val raw = if (shift < 0) {
                                    // Rec.601 luma: 0.299 R + 0.587 G + 0.114 B
                                    val r = (pixel shr 16) and 0xFF
                                    val g = (pixel shr 8) and 0xFF
                                    val b = pixel and 0xFF
                                    ((r * 76 + g * 150 + b * 30) ushr 8).coerceIn(0, 255)
                                } else (pixel shr shift) and 0xFF
                                currentInputBuffer.putFloat(norm(raw))
                            }
                        }
                        repeat(paddingFrames * pixelsPerImage) {
                            currentInputBuffer.putFloat(0f)
                        }
                    }
                }
                InputLayout.NTCHW -> {
                    // Layout: for each frame T, write all channels' H×W pixels in
                    // channel-major order. For grayscale (C=1) this is identical
                    // byte order to NTHWC. For RGB (C=3) we write all R values
                    // first, then all G, then all B for each frame.
                    for (px in pixelsPerFrame) {
                        for (c in 0 until numChannels) {
                            val shift = if (numChannels == 1) 16 else (2 - c) * 8
                            for (pixel in px) {
                                currentInputBuffer.putFloat(norm((pixel shr shift) and 0xFF))
                            }
                        }
                    }
                    val padFloats = paddingFrames * pixelsPerImage * numChannels
                    repeat(padFloats) { currentInputBuffer.putFloat(0f) }
                }
            }

            currentInputBuffer.rewind()

            // Diagnostic: log mean pixel value of frame 0 to confirm input varies across batches
            if (BuildConfig.DEBUG) {
                run {
                    val slice = FloatArray(inputHeight * inputWidth * numChannels)
                    currentInputBuffer.asFloatBuffer().get(slice)
                    currentInputBuffer.rewind()
                    val mean = slice.average()
                    Log.d("VSRInput", "input buffer frame0 mean=%.4f (%.1f/255)".format(mean, mean * 255))
                }
            }

            // 2. Prepare Output Buffer (already allocated)
            // Use the concretized output shape -- the model's declared shape
            // has -1 for batch and time; we substituted batch=1 and a
            // time-upper-bound = numFrames earlier. The decoder reads
            // [batch, T_out, vocab] floats out of the output buffer, so
            // these must be positive concrete values.
            val batchSize = concreteOutputShape[0]
            val timeSteps = concreteOutputShape[1]
            val vocabSize = concreteOutputShape[2]
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

            // Refresh from the engine: ONNX models with dynamic T_out report
            // the concrete shape only after the tensor has been produced.
            // OnnxModelEngine.run() updates its cached output shape during
            // run(); read it back here so the decoder loop iterates over the
            // *actual* timesteps the model emitted (not the upper bound).
            val actualOutputShape = engine.getOutputShape(0)
            val actualTimeSteps = if (actualOutputShape.size >= 2 && actualOutputShape[1] > 0)
                actualOutputShape[1] else timeSteps

            currentOutputBuffer.rewind()

            // 4. Decode
            // CTC models output raw logits. Apply softmax per timestep so the
            // BeamSearchDecoder receives proper probabilities in [0,1].
            // Iterate over the *actual* T_out from the engine, not the
            // upper-bound timeSteps used for buffer allocation.
            val probabilities = Array(actualTimeSteps) {
                val logits = FloatArray(vocabSize) { currentOutputBuffer.float }
                if (it == 0) {
                    Log.d("VSRInference", "logits[0] sample (first 5): ${logits.take(5)} actualT=$actualTimeSteps")
                }
                softmax(logits)
            }

            val decodedText = customDecoder?.invoke(probabilities)
                ?: if (useBeamSearch) beamDecoder.decode(probabilities) else greedyDecoder.decode(probabilities)

            // Compute confidence as mean max-softmax probability across all timesteps.
            // This reflects how "peaked" the model's output distribution was — a well-trained
            // model that recognised a clear phoneme will produce high max-prob values.
            val confidence = if (probabilities.isNotEmpty()) {
                probabilities.map { timeStep -> timeStep.maxOrNull() ?: 0f }.average().toFloat()
            } else 0f

            // Assign the same confidence to every word produced by this inference window.
            val wordConfidences = if (decodedText.isBlank()) {
                emptyList()
            } else {
                val wordCount = decodedText.trim().split("\\s+".toRegex()).size
                List(wordCount) { confidence }
            }

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
        val max = logits.maxOrNull() ?: 0f
        val exps = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }
}
