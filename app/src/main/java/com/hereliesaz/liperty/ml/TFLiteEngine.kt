package com.hereliesaz.liperty.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption

/**
 * ModelEngine backed by the TFLite/LiteRT Interpreter API.
 *
 * Using Interpreter (not CompiledModel) because the VSR model contains FlexErf
 * (a TF Select op).  CompiledModel has no Flex-delegate support; Interpreter
 * auto-links the Flex delegate when litert-select-tf-ops is on the classpath.
 *
 * Delegate priority: GPU → CPU (4 threads).
 */
class TFLiteEngine(
    private val context: Context,
    private val modelName: String = "vallr_model.tflite"
) : ModelEngine {

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var useInternalStorage = false

    fun setUseInternalStorage(use: Boolean) {
        useInternalStorage = use
    }

    override fun initialize(): Boolean {
        if (interpreter != null) return true

        return try {
            val buf = loadModelBuffer()
            interpreter = tryGpu(buf) ?: tryCpu(buf)
            Log.i("TFLiteEngine", "Model '$modelName' loaded successfully")
            true
        } catch (e: Exception) {
            Log.e("TFLiteEngine", "FAILED to load model '$modelName': ${e.message}")
            false
        }
    }

    // ── Buffer loading ────────────────────────────────────────────────────────

    private fun loadModelBuffer(): ByteBuffer {
        if (useInternalStorage) {
            val file = java.io.File(context.filesDir, modelName)
            if (file.exists()) {
                return FileChannel.open(file.toPath(), StandardOpenOption.READ)
                    .use { it.map(FileChannel.MapMode.READ_ONLY, 0, file.length()) }
            }
            Log.w("TFLiteEngine", "Internal file not found, falling back to assets")
        }
        return context.assets.openFd(modelName).use { fd ->
            fd.createInputStream().channel
                .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.length)
        }
    }

    // ── Interpreter creation ──────────────────────────────────────────────────

    private fun tryGpu(buf: ByteBuffer): Interpreter? {
        return try {
            val compat = CompatibilityList()
            if (!compat.isDelegateSupportedOnThisDevice) return null
            val delegate = GpuDelegate(compat.bestOptionsForThisDevice)
            val opts = Interpreter.Options().addDelegate(delegate)
            Interpreter(buf.duplicate(), opts).also {
                gpuDelegate = delegate
                Log.i("TFLiteEngine", "GPU delegate active for '$modelName'")
            }
        } catch (e: Exception) {
            Log.w("TFLiteEngine", "GPU delegate unavailable, falling back to CPU: ${e.message}")
            gpuDelegate?.close()
            gpuDelegate = null
            null
        }
    }

    private fun tryCpu(buf: ByteBuffer): Interpreter {
        val opts = Interpreter.Options().setNumThreads(4)
        return Interpreter(buf.duplicate(), opts).also {
            Log.i("TFLiteEngine", "CPU interpreter active for '$modelName'")
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
        val interp = interpreter
        if (interp == null) {
            Log.e("TFLiteEngine", "run() called but '$modelName' is not loaded — skipped")
            return
        }
        interp.run(inputBuffer, outputBuffer)
    }

    // ── Shape queries ─────────────────────────────────────────────────────────

    override fun getOutputShape(outputIndex: Int): IntArray =
        interpreter?.getOutputTensor(outputIndex)?.shape()
            ?: intArrayOf(1, 50, 40)

    override fun getInputShape(inputIndex: Int): IntArray =
        interpreter?.getInputTensor(inputIndex)?.shape()
            ?: intArrayOf(1, 50, 88, 88, 1)

    // ── Cleanup ───────────────────────────────────────────────────────────────

    override fun close() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
    }
}
