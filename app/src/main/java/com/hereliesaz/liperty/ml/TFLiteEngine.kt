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
            fun createModel(accelerator: Accelerator): CompiledModel {
                val options = CompiledModel.Options(accelerator)
                return if (useInternalStorage) {
                    val file = java.io.File(context.filesDir, modelName)
                    if (file.exists()) {
                        CompiledModel.create(file.absolutePath, options)
                    } else {
                        Log.w("TFLiteEngine", "Personalized model not found, falling back to assets: $modelName")
                        CompiledModel.create(context.assets, modelName, options)
                    }
                } else {
                    CompiledModel.create(context.assets, modelName, options)
                }
            }

            compiledModel = try {
                createModel(Accelerator.GPU)
            } catch (e: Exception) {
                Log.w("TFLiteEngine", "GPU compilation failed, falling back to CPU", e)
                createModel(Accelerator.CPU)
            }

            Log.i("TFLiteEngine", "LiteRT Model $modelName loaded successfully (Internal: $useInternalStorage)")
            return true
        } catch (e: Exception) {
            Log.e("TFLiteEngine", "FAILED to load model '$modelName': ${e.message}")
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
        // Model output: [batch=1, timeSteps=50, vocabSize=40]
        // Matches PHONEME_VOCAB (40 entries: 1 blank + 39 phonemes)
        return intArrayOf(1, 50, 40)
    }

    override fun getInputShape(inputIndex: Int): IntArray {
        // Model input: [batch=1, frames=50, height=88, width=88, channels=1]
        return intArrayOf(1, 50, 88, 88, 1)
    }

    override fun close() {
        compiledModel?.close()
    }
}
