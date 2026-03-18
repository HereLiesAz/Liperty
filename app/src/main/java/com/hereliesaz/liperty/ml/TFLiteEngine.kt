package com.hereliesaz.liperty.ml

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.TensorBuffer
import java.nio.ByteBuffer

class TFLiteEngine(
    private val context: Context,
    private val modelName: String = "vallr_model.tflite"
) : ModelEngine {

    private var compiledModel: CompiledModel? = null
    private var useInternalStorage = false

    fun setUseInternalStorage(use: Boolean) {
        this.useInternalStorage = use
    }

    override fun initialize(): Boolean {
        if (compiledModel != null) return true

        try {
            val options = try {
                CompiledModel.Options(Accelerator.GPU)
            } catch (e: Exception) {
                Log.e("TFLiteEngine", "GPU Accelerator not supported, falling back to CPU", e)
                CompiledModel.Options(Accelerator.CPU)
            }

            compiledModel = if (useInternalStorage) {
                val file = java.io.File(context.filesDir, modelName)
                if (file.exists()) {
                    CompiledModel.create(file.absolutePath, options)
                } else {
                    Log.w("TFLiteEngine", "Personalized model not found in internal storage, falling back to assets: $modelName")
                    CompiledModel.create(context.assets, modelName, options)
                }
            } else {
                CompiledModel.create(context.assets, modelName, options)
            }

            Log.i("TFLiteEngine", "LiteRT Model $modelName loaded successfully (Internal: $useInternalStorage)")
            return true
        } catch (e: Exception) {
            Log.e("TFLiteEngine", "Error initializing LiteRT", e)
            return false
        }
    }

    override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
        val model = compiledModel ?: return
        val inputBuffers = model.createInputBuffers()
        val outputBuffers = model.createOutputBuffers()

        // LiteRT 2.x TensorBuffer doesn't have loadBuffer(ByteBuffer) directly in some versions
        // Copy data from inputBuffer to FloatArray
        val inputSize = inputBuffer.remaining() / 4
        val inputArray = FloatArray(inputSize)
        inputBuffer.asFloatBuffer().get(inputArray)
        inputBuffers[0].writeFloat(inputArray)

        model.run(inputBuffers, outputBuffers)

        // Copy output back to outputBuffer
        val outputArray = outputBuffers[0].readFloat()
        outputBuffer.asFloatBuffer().put(outputArray)
    }

    override fun getOutputShape(outputIndex: Int): IntArray {
        // Return a mock shape for now as the new API makes it hard to query by index without names
        return intArrayOf(1, 16, 39)
    }

    override fun getInputShape(inputIndex: Int): IntArray {
        return intArrayOf(1, 16, 128)
    }

    override fun close() {
        compiledModel?.close()
    }
}
