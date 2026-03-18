package com.hereliesaz.liperty.ml

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Delegate
import com.google.ai.edge.litert.Interpreter
import com.google.ai.edge.litert.gpu.GpuDelegate
import com.google.ai.edge.litert.support.common.FileUtil
import java.nio.ByteBuffer

/**
 * Decodes non-auditory physiological signals (e.g., from BCMs) into text tokens.
 */
class SSREngine(private val context: Context) : ModelEngine {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: Delegate? = null
    private val MODEL_NAME = "ssr_model.tflite"

    override fun initialize(): Boolean {
        if (interpreter != null) return true
        try {
            val options = Interpreter.Options()
            try {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
            } catch (e: Exception) {
                Log.e("SSREngine", "GPU Delegate not supported, falling back to CPU", e)
            }
            // Attempt to load. If it fails, it will throw. The caller should handle it.
            val modelFile = FileUtil.loadMappedFile(context, MODEL_NAME)
            interpreter = Interpreter(modelFile, options)
            Log.i("SSREngine", "SSR Model loaded successfully")
            return true
        } catch (e: Exception) {
            Log.e("SSREngine", "Error initializing SSREngine", e)
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
