package com.hereliesaz.liperty.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/**
 * ModelEngine backed by the TFLite/LiteRT Interpreter API.
 *
 * Using Interpreter (not CompiledModel) because the VSR model contains FlexErf
 * (a TF Select op). CompiledModel has no Flex-delegate support; Interpreter
 * auto-links the Flex delegate when litert-select-tf-ops is on the classpath.
 *
 * GPU delegate is omitted for now: litert:2.1.3 and litert-gpu:1.4.2 ship
 * incompatible Delegate interfaces (org.tensorflow.lite.Delegate vs the GPU
 * module's shim), causing a compile error. CPU inference works correctly and
 * FlexErf runs via the auto-registered Flex delegate.
 */
class TFLiteEngine(
    private val context: Context,
    private val modelName: String = "vallr_model.tflite"
) : ModelEngine {

    private var interpreter: Interpreter? = null
    private var useInternalStorage = false

    fun setUseInternalStorage(use: Boolean) {
        useInternalStorage = use
    }

    override fun initialize(): Boolean {
        if (interpreter != null) return true

        return try {
            val buf = loadModelBuffer()
            val opts = Interpreter.Options().setNumThreads(4)
            interpreter = Interpreter(buf, opts)
            Log.i("TFLiteEngine", "Model '$modelName' loaded on CPU (4 threads)")
            true
        } catch (e: Exception) {
            Log.e("TFLiteEngine", "FAILED to load model '$modelName': ${e.message}")
            false
        }
    }

    private fun loadModelBuffer(): ByteBuffer {
        if (useInternalStorage) {
            val file = java.io.File(context.filesDir, modelName)
            if (file.exists()) {
                return FileInputStream(file).channel
                    .map(FileChannel.MapMode.READ_ONLY, 0, file.length())
                    .also { Log.i("TFLiteEngine", "Loaded '$modelName' from filesDir") }
            }
            Log.w("TFLiteEngine", "Internal file '$modelName' not found, using assets")
        }
        return context.assets.openFd(modelName).use { fd ->
            fd.createInputStream().channel
                .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.length)
                .also { Log.i("TFLiteEngine", "Loaded '$modelName' from assets") }
        }
    }

    override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
        val interp = interpreter
        if (interp == null) {
            Log.e("TFLiteEngine", "run() called but '$modelName' is not loaded — skipped")
            return
        }
        interp.run(inputBuffer, outputBuffer)
    }

    override fun getOutputShape(outputIndex: Int): IntArray =
        interpreter?.getOutputTensor(outputIndex)?.shape()
            ?: intArrayOf(1, 50, 40)

    override fun getInputShape(inputIndex: Int): IntArray =
        interpreter?.getInputTensor(inputIndex)?.shape()
            ?: intArrayOf(1, 50, 88, 88, 1)

    override fun close() {
        interpreter?.close()
        interpreter = null
    }
}
