package com.hereliesaz.liperty.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

/**
 * ONNX Runtime session wrapper for the AV-HuBERT V3 visual encoder.
 *
 * Distinct from [OnnxModelEngine] because the encoder has a *dynamic*
 * time dimension on its input (`(1, 1, T, 88, 88)`) and on its output
 * (`(1, T_out, 768)`). The shape-via-metadata path in [OnnxModelEngine]
 * stores -1 for those dims and then trips up on buffer-sizing.
 *
 * Output features are returned with their runtime shape so the decoder
 * step can feed them as `(B, T_enc, C)` to the cross-attention.
 */
/** Encoder side of the V3 backend — exposed as an interface so the
 *  orchestrator can be unit-tested with deterministic fakes. */
interface EncoderSession {
    fun initialize(): Boolean
    /** input: NCTHW `(1, 1, T, H, W)` float32 row-major;
     *  returns features `(B, T_out, C)` flattened, plus its runtime shape. */
    fun runEncode(input: FloatArray, inputShape: LongArray): Pair<FloatArray, LongArray>
    fun close()
}

class AvHubertEncoderSession(
    private val context: Context,
    private val modelName: String = "avhubert_base_vox_433h_visual_encoder.onnx",
) : EncoderSession {

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputName: String? = null
    private var outputName: String? = null

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
            inputName = s.inputInfo.keys.firstOrNull()
                ?: error("encoder ONNX has no inputs")
            outputName = s.outputInfo.keys.firstOrNull { it == "features" }
                ?: s.outputInfo.keys.first()
            env = e
            session = s
            Log.i(TAG, "Encoder '$modelName' loaded; input='$inputName' output='$outputName'")
            true
        } catch (e: Exception) {
            Log.e(TAG, "FAILED to load encoder '$modelName': ${e.message}", e)
            false
        }
    }

    /**
     * Run the encoder.
     *
     * @param input flat NCTHW row-major buffer of shape `(1, 1, T, 88, 88)`
     * @param inputShape that shape as `LongArray` for ONNX
     * @return `Pair(featuresFlat, featuresShape)` — featuresShape is
     *   `[1, T_out, 768]` reported by the ONNX runtime at run time.
     */
    override fun runEncode(input: FloatArray, inputShape: LongArray): Pair<FloatArray, LongArray> {
        val s = session ?: error("AvHubertEncoderSession not initialized")
        val e = env ?: error("AvHubertEncoderSession not initialized")
        require(inputShape.size == 5) { "input shape must be 5-D NCTHW, got ${inputShape.toList()}" }
        require(inputShape[0] == 1L) { "batch >1 not supported on-device" }

        val buf = FloatBuffer.wrap(input)
        OnnxTensor.createTensor(e, buf, inputShape).use { inputTensor ->
            s.run(mapOf(inputName!! to inputTensor)).use { results ->
                val out = results[0] as OnnxTensor
                val outShape = (out.info as TensorInfo).shape
                val feat = FloatArray((outShape[0] * outShape[1] * outShape[2]).toInt())
                out.floatBuffer.get(feat)
                return feat to outShape
            }
        }
    }

    private fun ensureModelOnDisk(): String {
        val dst = File(context.filesDir, modelName)
        val assetSize = context.assets.openFd(modelName).use { it.length }
        if (dst.exists() && dst.length() == assetSize) return dst.absolutePath
        context.assets.open(modelName).use { input ->
            dst.outputStream().use { output -> input.copyTo(output) }
        }
        Log.i(TAG, "Materialised encoder model to ${dst.absolutePath} (${dst.length()} bytes)")
        return dst.absolutePath
    }

    override fun close() {
        session?.close()
        session = null
        env = null
    }

    companion object {
        private const val TAG = "AvHubertEncoderSession"
    }
}
