package com.hereliesaz.liperty.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log

/**
 * Encoder + autoregressive Transformer decoder orchestrator. Originally
 * authored for the V3 AV-HuBERT seq2seq backend ([create]); now also
 * drives the SyncVSR seq2seq path ([createSyncVsr]) since both share
 * the same contract: 5-D grayscale clip input, encoder produces
 * (B, T_enc, D) hidden states, attention decoder consumes them and the
 * running prev-token history step-by-step.
 *
 * Parallel to [VSRInference] (the V2/Auto-AVSR + SyncVSR CTC path), not
 * a replacement. Selection happens at the caller site.
 *
 * One call to [runInference] runs:
 *   1. Crop + normalize the last [numFrames] frames into a flat buffer
 *      whose shape header matches [inputLayout] (NCTHW for AV-HuBERT,
 *      NTCHW for SyncVSR).
 *   2. Encoder ONNX -> `(1, T_enc, D)` features.
 *   3. Greedy autoregressive decode through the decoder ONNX, building
 *      a token sequence one BPE/unigram token at a time, stopping on
 *      EOS or at [maxDecodeSteps].
 *   4. Detokenize via [BpeDetokenizer].
 *
 * Beam search is NOT here yet. Greedy is the baseline; revisit once
 * device-side WER is measured.
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
    /**
     * Input layout for the encoder ONNX. AV-HuBERT was exported NCTHW
     * (the default); SyncVSR was exported NTCHW. For grayscale (C=1) the
     * flat row-major byte order is identical between the two — only the
     * declared shape on the ONNX input tensor differs. Pass null to keep
     * back-compat NCTHW; pass [InputLayout.NTCHW] for the SyncVSR encoder.
     */
    inputLayout: InputLayout = InputLayout.NCTHW,
) {

    private val greedy = Seq2SeqGreedyDecoder(bosId, eosId, maxDecodeSteps)

    private val inputElements = 1 * 1 * numFrames * cropSize * cropSize
    private val inputShape: LongArray = when (inputLayout) {
        InputLayout.NCTHW -> longArrayOf(1L, 1L, numFrames.toLong(), cropSize.toLong(), cropSize.toLong())
        InputLayout.NTCHW -> longArrayOf(1L, numFrames.toLong(), 1L, cropSize.toLong(), cropSize.toLong())
        InputLayout.NTHWC -> longArrayOf(1L, numFrames.toLong(), cropSize.toLong(), cropSize.toLong(), 1L)
    }

    fun initialize(): Boolean {
        val a = encoder.initialize()
        val b = decoder.initialize()
        return a && b
    }

    /** True only when both ONNX sessions are loaded and [runInference] is safe.
     *  The caller must check this before routing frames here — a constructed but
     *  uninitialized backend (e.g. its model file was never downloaded) would
     *  otherwise throw inside [runInference] and crash the inference coroutine. */
    fun isReady(): Boolean = encoder.isReady() && decoder.isReady()

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
     * Bitmap -> flat FloatArray with grayscale + mean/std normalization,
     * shape declared at construction time (NCTHW for AV-HuBERT, NTCHW for
     * SyncVSR). For C=1 the on-disk byte order is identical between the
     * two layouts; only the shape header on the ONNX input tensor changes.
     *
     * The mouth ROI alignment (face-relative crop, 88x88 resize) happens
     * upstream in MainActivity / FaceLandmarkerHelper. This function just
     * normalizes whatever bitmaps it gets.
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

        // For C=1 grayscale, NCTHW and NTCHW flat write orders coincide:
        // both iterate T outer, then H*W inner (the C=1 dim contributes
        // no extra steps either way). So this loop is layout-agnostic
        // when channels=1, which is the only shape the seq2seq encoders
        // (AV-HuBERT, SyncVSR) accept.
        // Grayscale via Rec.601 luma: Y = (76R + 150G + 30B) / 256. The
        // training pipeline (Chaplin transforms.py) reads .mp4 with
        // PIL/cv2 which both default to Y'CbCr-derived grayscale, NOT
        // raw red channel. Using just R drifts the mean and skews
        // skin-tone pixels.
        var w = 0
        for (px in pixels) {
            for (pixel in px) {
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                val raw = ((r * 76 + g * 150 + b * 30) ushr 8).coerceIn(0, 255)
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
            val dictText = VocabularyLoader.readTextPreferringFiles(context, dictAsset)
            val dict = VocabularyLoader.parseTokenList(dictText)
            return AvHubertSeq2SeqInference(enc, dec, dict)
        }

        /**
         * SyncVSR seq2seq factory. Wires `syncvsr_lrs3_encoder.onnx`
         * (NTCHW, hidden states with no CTC head) + `syncvsr_lrs3_decoder.onnx`
         * (espnet attention decoder, no KV cache) + the SyncVSR unigram
         * vocab (5049 tokens; blank=0, eos=5048; espnet reuses eos as sos).
         *
         * Mouth-ROI preprocessing is identical to the SyncVSR CTC path
         * (88x88, mean=0.421, std=0.165, grayscale via Rec.601 luma) so
         * the camera pipeline upstream of this class doesn't change —
         * MainActivity picks whether to route the frame window through
         * VSRInference (CTC) or here (attention decoder).
         *
         * Window: 16 frames to match the active FrameBuffer capacity.
         * Max decode steps: 64 (room for ~12-15 words of SentencePiece
         * unigram output before bailing out of the autoregressive loop).
         */
        fun createSyncVsr(
            context: Context,
            encoderAsset: String = "syncvsr_lrs3_encoder.onnx",
            decoderAsset: String = "syncvsr_lrs3_decoder.onnx",
            vocabAsset: String = "syncvsr_unigram_units.txt",
            numFrames: Int = 16,
        ): AvHubertSeq2SeqInference {
            val enc = AvHubertEncoderSession(context, encoderAsset)
            val dec = AvHubertDecoderSession(context, decoderAsset)
            val vocabText = VocabularyLoader.readTextPreferringFiles(context, vocabAsset)
            val vocab = VocabularyLoader.parseTokenList(vocabText)
            // espnet convention: <eos> doubles as <sos>. SyncVSR's unigram
            // dump ends with <eos> at index (vocab.size - 1).
            val eosId = vocab.size - 1
            return AvHubertSeq2SeqInference(
                encoder = enc,
                decoder = dec,
                dict = vocab,
                bosId = eosId,
                eosId = eosId,
                numFrames = numFrames,
                cropSize = 88,
                pixelMean = 0.421f,
                pixelStd = 0.165f,
                maxDecodeSteps = 64,
                inputLayout = InputLayout.NTCHW,
            )
        }
    }
}
