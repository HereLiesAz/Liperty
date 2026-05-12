package com.hereliesaz.liperty.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log

/**
 * V3 inference orchestrator: AV-HuBERT encoder + Transformer decoder
 * (seq2seq autoregressive) + SentencePiece BPE detokenization.
 *
 * Parallel to [VSRInference] (the V2/Auto-AVSR CTC path), not a
 * replacement. Selection between V2 and V3 happens at the caller site
 * (currently a hardcoded path in `MainActivity`; eventually a setting).
 *
 * One call to [runInference] runs:
 *   1. Crop + normalize the last 50 frames to `(1, 1, 50, 88, 88)`.
 *   2. Encoder ONNX -> `(1, T_enc, 768)` features.
 *   3. Greedy autoregressive decode through the decoder ONNX, building
 *      a token sequence one BPE token at a time, stopping on EOS or at
 *      [maxDecodeSteps].
 *   4. Detokenize via [BpeDetokenizer].
 *
 * Beam search is NOT here yet. Greedy is the baseline; revisit once
 * device-side WER is measured (Phase 4).
 */
class AvHubertSeq2SeqInference(
    private val encoder: EncoderSession,
    private val decoder: DecoderSession,
    private val dict: List<String>,
    private val bosId: Int = 0,
    private val eosId: Int = 2,
    private val numFrames: Int = 50,
    private val cropSize: Int = 88,
    private val pixelMean: Float = 0.421f,
    private val pixelStd: Float = 0.165f,
    private val maxDecodeSteps: Int = 50,
) {

    private val greedy = Seq2SeqGreedyDecoder(bosId, eosId, maxDecodeSteps)

    private val inputElements = 1 * 1 * numFrames * cropSize * cropSize
    private val inputShape = longArrayOf(1L, 1L, numFrames.toLong(), cropSize.toLong(), cropSize.toLong())

    fun initialize(): Boolean {
        val a = encoder.initialize()
        val b = decoder.initialize()
        return a && b
    }

    /**
     * Run V3 inference on the supplied frames. Takes the LAST [numFrames]
     * frames if more are provided; pads with zeros if fewer.
     */
    @Synchronized
    fun runInference(frames: List<Bitmap>): VSRResult {
        val t0 = SystemClock.uptimeMillis()
        val input = preprocessFrames(frames)
        val (feats, featShape) = encoder.runEncode(input, inputShape)
        val tokens = greedy.decode { prev ->
            decoder.runStep(prev, feats, featShape)
        }
        val text = BpeDetokenizer.detokenize(tokens, dict)
        val dt = SystemClock.uptimeMillis() - t0
        Log.i(TAG, "V3 inference: ${tokens.size} tokens, ${dt} ms, text='$text'")
        return VSRResult(text = text, confidence = 1f, processingTimeMs = dt)
    }

    /**
     * Bitmap -> (1, 1, T, 88, 88) flat FloatArray with center mouth crop +
     * AV-HuBERT mean/std normalization. NCTHW layout: C-major, then T,
     * then H*W row-major.
     *
     * The mouth ROI math is intentionally identical to the V2 path:
     * grayscale (red channel), centered ~2/3 down the face, 112x112 then
     * resized to 88x88. Mediapipe-driven precise alignment is the camera
     * pipeline's job upstream of this class; here we just trust the
     * crop already happened.
     */
    private fun preprocessFrames(frames: List<Bitmap>): FloatArray {
        val out = FloatArray(inputElements)
        val take = if (frames.size > numFrames) frames.takeLast(numFrames) else frames
        val pixelsPerImage = cropSize * cropSize

        // Decode each frame to a single int[] of ARGB pixels at cropSize x cropSize.
        val pixels = take.map { bmp ->
            val scaled = if (bmp.width != cropSize || bmp.height != cropSize) {
                Bitmap.createScaledBitmap(bmp, cropSize, cropSize, true)
            } else bmp
            val px = IntArray(pixelsPerImage)
            scaled.getPixels(px, 0, cropSize, 0, 0, cropSize, cropSize)
            if (scaled !== bmp) scaled.recycle()
            px
        }

        // NCTHW write order: for each channel (C=1 -> red only), for each frame T,
        // for each pixel H*W: write (px/255 - mean) / std.
        var w = 0
        // channel 0 = red (grayscale proxy, matches V2's Auto-AVSR convention)
        for (px in pixels) {
            for (pixel in px) {
                val raw = (pixel shr 16) and 0xFF
                out[w++] = (raw / 255f - pixelMean) / pixelStd
            }
        }
        val pad = (numFrames - pixels.size) * pixelsPerImage
        // Remaining frames in the buffer are already 0.0 (default init), but
        // be explicit so the intent is obvious and we don't depend on it.
        repeat(pad) { out[w++] = 0f }
        return out
    }

    fun close() {
        encoder.close()
        decoder.close()
    }

    companion object {
        private const val TAG = "AvHubertSeq2Seq"

        /** Convenience constructor that builds the encoder, decoder, and
         *  loads the dictionary from app assets. */
        fun create(
            context: Context,
            encoderAsset: String = "avhubert_base_vox_433h_visual_encoder.onnx",
            decoderAsset: String = "avhubert_base_vox_433h_decoder.onnx",
            dictAsset: String = "avhubert_base_vox_433h_dict.txt",
        ): AvHubertSeq2SeqInference {
            val enc = AvHubertEncoderSession(context, encoderAsset)
            val dec = AvHubertDecoderSession(context, decoderAsset)
            val dictText = context.assets.open(dictAsset).bufferedReader().use { it.readText() }
            val dict = VocabularyLoader.parseTokenList(dictText)
            return AvHubertSeq2SeqInference(enc, dec, dict)
        }
    }
}
