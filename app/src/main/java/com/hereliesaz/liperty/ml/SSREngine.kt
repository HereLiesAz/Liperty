package com.hereliesaz.liperty.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer

/**
 * Decodes non-auditory physiological signals (e.g., from BCMs) into text tokens.
 */
class SSREngine(private val context: Context) : ModelEngine {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: Delegate? = null
    private val MODEL_NAME = "ssr_model.tflite"

    override fun initialize() {
        if (interpreter != null) return
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
        } catch (e: Exception) {
            Log.e("SSREngine", "Error initializing SSREngine", e)
        }
    }

    override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
        interpreter?.run(inputBuffer, outputBuffer)
    }

    override fun getOutputShape(outputIndex: Int): IntArray {
        return interpreter?.getOutputTensor(outputIndex)?.shape() ?: IntArray(0)
    }

    override fun close() {
        interpreter?.close()
        gpuDelegate?.close()
    }
}
