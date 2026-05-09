package com.hereliesaz.liperty.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.FloatBuffer

/**
 * ModelEngine backed by ONNX Runtime. Built for the VALLR VideoMAE → Wav2Vec2-CTC
 * checkpoint exported from PyTorch, which expects NCTHW input
 * (1, 3, 16, 224, 224) and produces (1, 8, 40) phoneme logits.
 *
 * Loads the .onnx file from app assets — bundled at build time.
 */
class OnnxModelEngine(
    private val context: Context,
    private val modelName: String = "vallr_model.onnx"
) : ModelEngine {

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputName: String? = null
    private var inputShape: IntArray = intArrayOf()
    private var outputShape: IntArray = intArrayOf()

    @Synchronized
    override fun initialize(): Boolean {
        if (session != null) return true
        return try {
            val modelPath = ensureModelOnDisk()
            val e = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val s = e.createSession(modelPath, opts)

            // Cache I/O metadata so we don't reflect on every inference call.
            val inputInfo = s.inputInfo.entries.first()
            inputName = inputInfo.key
            inputShape = (inputInfo.value.info as ai.onnxruntime.TensorInfo)
                .shape.map { it.toInt() }.toIntArray()
            val outputInfo = s.outputInfo.entries.first()
            outputShape = (outputInfo.value.info as ai.onnxruntime.TensorInfo)
                .shape.map { it.toInt() }.toIntArray()

            env = e
            session = s
            Log.i(
                "OnnxModelEngine",
                "Model '$modelName' loaded. input='$inputName' shape=${inputShape.contentToString()} " +
                    "output shape=${outputShape.contentToString()}"
            )
            true
        } catch (e: Exception) {
            Log.e("OnnxModelEngine", "FAILED to load model '$modelName': ${e.message}", e)
            false
        }
    }

    override fun run(inputBuffer: ByteBuffer, outputBuffer: ByteBuffer) {
        val s = session ?: run {
            Log.e("OnnxModelEngine", "run() called but '$modelName' is not loaded — skipped")
            return
        }
        val name = inputName ?: return
        val e = env ?: return

        // Caller writes input in the layout reported by getInputLayout() = NCTHW,
        // so we can hand the FloatBuffer directly to ONNX with the model's shape.
        inputBuffer.rewind()
        val floatBuf: FloatBuffer = inputBuffer.asFloatBuffer()
        val longShape = inputShape.map { it.toLong() }.toLongArray()

        OnnxTensor.createTensor(e, floatBuf, longShape).use { input ->
            s.run(mapOf(name to input)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val out = results[0] as OnnxTensor
                copyTensorTo(out, outputBuffer)
            }
        }
    }

    private fun copyTensorTo(tensor: OnnxTensor, dst: ByteBuffer) {
        dst.rewind()
        dst.asFloatBuffer().put(tensor.floatBuffer)
    }

    /**
     * Streams the bundled ONNX asset into [Context.getFilesDir] so that
     * [OrtEnvironment.createSession] can memory-map it. Avoids loading the
     * (~357 MB) model into the Java heap, which would risk OOM on
     * memory-constrained devices.
     */
    private fun ensureModelOnDisk(): String {
        val dst = File(context.filesDir, modelName)
        val assetSize = context.assets.openFd(modelName).use { it.length }
        if (dst.exists() && dst.length() == assetSize) return dst.absolutePath

        context.assets.open(modelName).use { input ->
            dst.outputStream().use { output -> input.copyTo(output) }
        }
        Log.i("OnnxModelEngine", "Materialised model to ${dst.absolutePath} (${dst.length()} bytes)")
        return dst.absolutePath
    }

    override fun getOutputShape(outputIndex: Int): IntArray =
        if (outputIndex == 0 && outputShape.isNotEmpty()) outputShape
        else intArrayOf()

    override fun getInputShape(inputIndex: Int): IntArray =
        if (inputIndex == 0 && inputShape.isNotEmpty()) inputShape
        else intArrayOf()

    override fun getInputLayout(): InputLayout = InputLayout.NCTHW

    override fun close() {
        session?.close()
        session = null
        // OrtEnvironment is process-wide and cached internally; do not close it here.
        env = null
    }
}
