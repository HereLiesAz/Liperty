package com.hereliesaz.liperty.ml

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Delegate
import com.google.ai.edge.litert.Interpreter
import com.google.ai.edge.litert.gpu.GpuDelegate
import com.google.ai.edge.litert.support.common.FileUtil
import java.nio.ByteBuffer

class TFLiteEngine(
    private val context: Context,
    private val modelName: String = "vallr_model.tflite"
) : ModelEngine {

    private var interpreter: Interpreter? = null
    private var gpuDelegate: Delegate? = null
    private var useInternalStorage = false

    fun setUseInternalStorage(use: Boolean) {
        this.useInternalStorage = use
    }

    override fun initialize(): Boolean {
        if (interpreter != null) return true

        try {
            val options = Interpreter.Options()
            try {
                // Try to initialize GPU delegate. This might fail if native libs are missing.
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
            } catch (e: Throwable) {
                Log.e("TFLiteEngine", "GPU Delegate not supported, falling back to CPU", e)
            }

            val modelFile = if (useInternalStorage) {
                val file = java.io.File(context.filesDir, modelName)
                if (file.exists()) {
                    java.nio.channels.FileChannel.open(file.toPath(), java.nio.file.StandardOpenOption.READ).use { channel ->
                        channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, channel.size())
                    }
                } else {
                    Log.w("TFLiteEngine", "Personalized model not found in internal storage, falling back to assets: $modelName")
                    FileUtil.loadMappedFile(context, modelName)
                }
            } else {
                FileUtil.loadMappedFile(context, modelName)
            }

            interpreter = Interpreter(modelFile, options)
            Log.i("TFLiteEngine", "TFLite Model $modelName loaded successfully (Internal: $useInternalStorage)")
            return true
        } catch (e: java.io.FileNotFoundException) {
            Log.e("TFLiteEngine", "Model file not found: $modelName")
            return false
        } catch (e: Exception) {
            Log.e("TFLiteEngine", "Error initializing TFLite", e)
            return false
        }
    }

    override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
        interpreter?.run(inputBuffer, outputBuffer)
    }

    override fun getOutputShape(outputIndex: Int): IntArray {
        return interpreter?.getOutputTensor(outputIndex)?.shape() ?: IntArray(0)
    }

    override fun getInputShape(inputIndex: Int): IntArray {
        return interpreter?.getInputTensor(inputIndex)?.shape() ?: IntArray(0)
    }

    override fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }
}
