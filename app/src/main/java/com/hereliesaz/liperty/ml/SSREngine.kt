package com.hereliesaz.liperty.ml

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Accelerator
import java.nio.ByteBuffer

/**
 * Decodes non-auditory physiological signals (e.g., from BCMs) into text tokens.
 * Migrated to LiteRT 2.x CompiledModel API.
 */
class SSREngine(private val context: Context) : ModelEngine {
    private var compiledModel: CompiledModel? = null
    private val MODEL_NAME = "ssr_model.tflite"

    override fun initialize(): Boolean {
        if (compiledModel != null) return true
        try {
            val options = try {
                CompiledModel.Options(Accelerator.GPU)
            } catch (e: Exception) {
                Log.e("SSREngine", "GPU Accelerator not supported, falling back to CPU", e)
                CompiledModel.Options(Accelerator.CPU)
            }
            // Attempt to load.
            compiledModel = CompiledModel.create(context.assets, MODEL_NAME, options)
            Log.i("SSREngine", "SSR Model loaded successfully with LiteRT")
            return true
        } catch (e: Exception) {
            Log.e("SSREngine", "Error initializing SSREngine", e)
            return false
        }
    }

    override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
        val model = compiledModel ?: return
        val inputBuffers = model.createInputBuffers()
        val outputBuffers = model.createOutputBuffers()

        val inputSize = inputBuffer.remaining() / 4
        val inputArray = FloatArray(inputSize)
        inputBuffer.asFloatBuffer().get(inputArray)
        inputBuffers[0].writeFloat(inputArray)

        model.run(inputBuffers, outputBuffers)

        val outputArray = outputBuffers[0].readFloat()
        outputBuffer.asFloatBuffer().put(outputArray)
    }

    override fun getOutputShape(outputIndex: Int): IntArray {
        return intArrayOf(1, 16, 39)
    }

    override fun getInputShape(inputIndex: Int): IntArray {
        return intArrayOf(1, 16, 128)
    }

    override fun close() {
        compiledModel?.close()
    }
}
