package com.hereliesaz.liperty.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * ONNX Runtime session wrapper for the AV-HuBERT V3 seq2seq decoder
 * (see `tools/kaggle_avhubert_export_decoder.py`).
 *
 * The decoder is exported as a NO-KV-cache single-step model: each call
 * re-runs the full transformer decoder over all previously emitted
 * tokens and returns logits for every position. We only care about the
 * last position (the next-token prediction), so `runStep` extracts and
 * returns just that `(V,)` slice — the orchestrator never sees the
 * intermediate `(1, T_dec, V)` tensor.
 *
 * Doesn't implement [ModelEngine] because that interface is single-input
 * + single-output, and this decoder takes two inputs.
 */
/** Decoder side of the V3 backend — exposed as an interface so the
 *  orchestrator can be unit-tested with deterministic fakes. */
interface DecoderSession {
    fun initialize(): Boolean
    fun runStep(prevTokens: IntArray, encoderFeatures: FloatArray, encShape: LongArray): FloatArray
    val vocabSize: Int
    fun close()
}

class AvHubertDecoderSession(
    private val context: Context,
    private val modelName: String = "avhubert_base_vox_433h_decoder.onnx",
) : DecoderSession {

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputPrevTokens: String? = null
    private var inputEncoderFeatures: String? = null
    private var outputLogits: String? = null

    /** Vocab size, surfaced for callers that build dictionaries / decoders. */
    override var vocabSize: Int = -1
        private set

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

            // Resolve named inputs/outputs by their declared role rather than
            // by position — the export script names them explicitly so this
            // is robust to ordering surprises in the ONNX graph.
            val inputs = s.inputInfo.keys
            inputPrevTokens = inputs.firstOrNull { it == "prev_tokens" }
                ?: error("decoder ONNX missing 'prev_tokens' input. Got: $inputs")
            inputEncoderFeatures = inputs.firstOrNull { it == "encoder_features" }
                ?: error("decoder ONNX missing 'encoder_features' input. Got: $inputs")
            val outputs = s.outputInfo.keys
            outputLogits = outputs.firstOrNull { it == "logits" }
                ?: error("decoder ONNX missing 'logits' output. Got: $outputs")

            // Logits output shape is (B, T_dec, V). V is the last (static) dim.
            val logitsShape = (s.outputInfo[outputLogits]!!.info as TensorInfo).shape
            vocabSize = logitsShape.last().toInt()

            env = e
            session = s
            Log.i(TAG, "Decoder '$modelName' loaded; vocab=$vocabSize")
            true
        } catch (e: Exception) {
            Log.e(TAG, "FAILED to load decoder '$modelName': ${e.message}", e)
            false
        }
    }

    /**
     * One autoregressive step. Returns the `(V,)` logits for the next
     * token; caller takes argmax (or beam-search expansion).
     *
     * @param prevTokens history including BOS, length `T_dec`
     * @param encoderFeatures flat row-major `(1, T_enc, C)` FloatArray
     * @param encShape `[1, T_enc, C]` shape of `encoderFeatures`
     */
    override fun runStep(
        prevTokens: IntArray,
        encoderFeatures: FloatArray,
        encShape: LongArray,
    ): FloatArray {
        val s = session ?: error("AvHubertDecoderSession not initialized")
        val e = env ?: error("AvHubertDecoderSession not initialized")
        require(encShape.size == 3) { "encShape must be [B, T_enc, C], got ${encShape.toList()}" }
        require(encShape[0] == 1L) { "batch >1 not supported on-device" }

        val tDec = prevTokens.size
        val tokLong = LongArray(tDec) { prevTokens[it].toLong() }
        val tokShape = longArrayOf(1L, tDec.toLong())

        val featBuf = FloatBuffer.wrap(encoderFeatures)
        val tokBuf = LongBuffer.wrap(tokLong)

        OnnxTensor.createTensor(e, tokBuf, tokShape).use { tokIn ->
            OnnxTensor.createTensor(e, featBuf, encShape).use { featIn ->
                val feeds = mapOf(
                    inputPrevTokens!! to tokIn,
                    inputEncoderFeatures!! to featIn,
                )
                s.run(feeds).use { results ->
                    val out = results[0] as OnnxTensor
                    return extractLastPositionLogits(out, tDec, vocabSize)
                }
            }
        }
    }

    private fun extractLastPositionLogits(out: OnnxTensor, tDec: Int, v: Int): FloatArray {
        // ONNX output is (1, T_dec, V) — pull the last V floats.
        val fb = out.floatBuffer
        val total = 1 * tDec * v
        check(fb.remaining() == total) {
            "Decoder output size mismatch: got ${fb.remaining()}, expected $total"
        }
        val result = FloatArray(v)
        // Position offset of the last token slice
        fb.position(total - v)
        fb.get(result)
        return result
    }

    private fun ensureModelOnDisk(): String {
        val dst = File(context.filesDir, modelName)
        val assetSize = context.assets.openFd(modelName).use { it.length }
        if (dst.exists() && dst.length() == assetSize) return dst.absolutePath
        context.assets.open(modelName).use { input ->
            dst.outputStream().use { output -> input.copyTo(output) }
        }
        Log.i(TAG, "Materialised decoder model to ${dst.absolutePath} (${dst.length()} bytes)")
        return dst.absolutePath
    }

    override fun close() {
        session?.close()
        session = null
        env = null
    }

    companion object {
        private const val TAG = "AvHubertDecoderSession"
    }
}
