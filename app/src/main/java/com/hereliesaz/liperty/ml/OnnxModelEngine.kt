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
    private val modelName: String = "vallr_model.onnx",
    /** Layout the loaded model expects for its primary input tensor.
     *  Auto-AVSR uses NCTHW; SyncVSR uses NTCHW. Callers pass whichever
     *  matches the bundled ONNX so VSRInference writes pixels in the
     *  right axis order. Default NCTHW keeps the historical behaviour
     *  (VALLR/Auto-AVSR exports). */
    private val expectedInputLayout: InputLayout = InputLayout.NCTHW,
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

        // Build a CONCRETE shape from the model's declared shape, which may
        // contain -1 placeholders for batch (N) and time (T) axes. ONNX
        // Runtime rejects negative dims when creating the input tensor, so
        // we substitute:
        //   - batch axis (-1): always 1 for our pipeline
        //   - exactly-one remaining dynamic axis (T): derive from the input
        //     buffer's float count divided by the product of the fixed dims
        // If multiple dynamic axes remain after batch, we'd need explicit
        // runtime info from the caller -- our current models (Auto-AVSR,
        // AV-HuBERT visual) only have N and T dynamic, so this works.
        val totalFloats = inputBuffer.capacity().toLong() / 4L
        val longShape = LongArray(inputShape.size) { i ->
            if (inputShape[i] > 0) inputShape[i].toLong()
            else if (i == 0) 1L  // batch
            else -1L             // placeholder, resolved below
        }
        // Resolve the remaining dynamic dim (typically T) from the buffer size
        // and the known fixed dims. There should be exactly one -1 left.
        var fixedProduct = 1L
        for (j in longShape.indices) if (longShape[j] > 0) fixedProduct *= longShape[j]
        for (j in longShape.indices) {
            if (longShape[j] == -1L) {
                longShape[j] = (totalFloats / fixedProduct).coerceAtLeast(1L)
                fixedProduct *= longShape[j]
            }
        }

        OnnxTensor.createTensor(e, floatBuf, longShape).use { input ->
            s.run(mapOf(name to input)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val out = results[0] as OnnxTensor
                // Refresh outputShape with the actual returned shape so the
                // decoder gets concrete dims (model's declared output shape
                // has -1 for batch and T_out).
                outputShape = out.info.shape.map { it.toInt() }.toIntArray()
                copyTensorTo(out, outputBuffer)
            }
        }
    }

    private fun copyTensorTo(tensor: OnnxTensor, dst: ByteBuffer) {
        dst.rewind()
        dst.asFloatBuffer().put(tensor.floatBuffer)
    }

    /**
     * Ensures the ONNX model is available on disk for memory-mapped loading.
     *
     * Checks (in order):
     * 1. **filesDir** — model already downloaded by [ModelDownloadManager]
     *    or copied from assets on a previous launch.
     * 2. **assets** — model bundled in APK (dev builds with setup_libs.sh).
     *    Copied to filesDir for ORT session compatibility.
     * 3. Neither — throws [IllegalStateException] (model must be downloaded
     *    via the setup screen first).
     */
    private fun ensureModelOnDisk(): String {
        val dst = File(context.filesDir, modelName)

        // 1. Already on disk (runtime download or previous asset copy)
        if (dst.exists() && dst.length() > 1000) return dst.absolutePath

        // 2. Try bundled assets (dev builds)
        try {
            val assetSize = context.assets.open(modelName).use { it.available().toLong() }
            if (assetSize > 1000) {
                context.assets.open(modelName).use { input ->
                    dst.outputStream().use { output -> input.copyTo(output) }
                }
                Log.i("OnnxModelEngine", "Copied model from assets to ${dst.absolutePath} (${dst.length()} bytes)")
                return dst.absolutePath
            }
        } catch (_: Exception) {
            // Asset not bundled — expected for production builds
        }

        // 3. Model not available
        throw IllegalStateException(
            "Model '$modelName' not found. Run initial setup to download models."
        )
    }

    override fun getOutputShape(outputIndex: Int): IntArray =
        if (outputIndex == 0 && outputShape.isNotEmpty()) outputShape
        else intArrayOf()

    override fun getInputShape(inputIndex: Int): IntArray =
        if (inputIndex == 0 && inputShape.isNotEmpty()) inputShape
        else intArrayOf()

    override fun getInputLayout(): InputLayout = expectedInputLayout

    override fun close() {
        session?.close()
        session = null
        // OrtEnvironment is process-wide and cached internally; do not close it here.
        env = null
    }
}
