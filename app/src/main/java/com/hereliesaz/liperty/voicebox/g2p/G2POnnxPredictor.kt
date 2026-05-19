package com.hereliesaz.liperty.voicebox.g2p

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer

/**
 * Neural G2P (grapheme-to-phoneme) predictor backed by an ONNX Runtime session.
 *
 * The model is a sequence-to-sequence network trained on the CMU Pronouncing
 * Dictionary. It maps a character sequence to an ARPABET phoneme sequence:
 *
 *     word chars → char indices (LongArray) → ONNX → phoneme indices → ARPABET strings
 *
 * Used as the out-of-vocabulary (OOV) fallback when a word is absent from
 * [CMUDictionary]. If the model is unavailable, [predict] returns an empty
 * list and [MeloTTSTokenizer] drops to the letter-by-letter rule fallback.
 *
 * ONNX Runtime usage follows the same pattern as [com.hereliesaz.liperty.voicebox.PocketTTSEngine]:
 * - [OrtEnvironment] is provided externally (process-wide singleton).
 * - All session calls are wrapped in try/catch — failures are silent.
 * - [close] nulls the session; [OrtEnvironment] is not closed here.
 */
class G2POnnxPredictor(private val ortEnv: OrtEnvironment) {

    private var session: OrtSession? = null

    private var graphemeVocab: List<String> = emptyList()
    private var phonemeVocab: List<String> = emptyList()

    // Derived lookup maps, rebuilt in initialize().
    private var graphemeToIndex: Map<String, Int> = emptyMap()
    private var indexToPhoneme: Map<Int, String> = emptyMap()

    /** True once [initialize] succeeded and the session is live. */
    val isLoaded: Boolean get() = session != null

    /**
     * Loads the ONNX session from an absolute file-system path and builds the
     * vocabulary lookup tables.
     *
     * @param modelPath   Absolute path to the G2P ONNX model file.
     * @param graphemeVocab List of grapheme (character) symbols in vocab-index order.
     *                      Index 0 is typically a padding or unknown token.
     * @param phonemeVocab  List of ARPABET phoneme symbols in vocab-index order.
     *                      Index 0 is typically a blank / EOS token.
     * @return `true` on success, `false` if loading fails for any reason.
     */
    fun initialize(
        modelPath: String,
        graphemeVocab: List<String>,
        phonemeVocab: List<String>
    ): Boolean {
        return try {
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            session = ortEnv.createSession(modelPath, opts)

            this.graphemeVocab = graphemeVocab
            this.phonemeVocab = phonemeVocab

            graphemeToIndex = graphemeVocab.withIndex().associate { (i, sym) -> sym to i }
            indexToPhoneme = phonemeVocab.withIndex().associate { (i, sym) -> i to sym }

            true
        } catch (e: Exception) {
            session = null
            false
        }
    }

    /**
     * Predicts the ARPABET phoneme sequence for [word].
     *
     * Steps:
     * 1. Lowercase [word] and split into individual characters.
     * 2. Map each character to its grapheme vocab index (unknown chars → 0).
     * 3. Run the ONNX session with a `(1, T_in)` int64 input tensor.
     * 4. Interpret the output as a `(T_out,)` or `(1, T_out)` sequence of
     *    phoneme vocab indices and map them back to ARPABET strings.
     * 5. Filter out blank / padding tokens (index 0 and empty strings).
     *
     * Returns an empty list on any failure.
     */
    fun predict(word: String): List<String> {
        val s = session ?: return emptyList()
        if (graphemeVocab.isEmpty() || phonemeVocab.isEmpty()) return emptyList()
        if (word.isBlank()) return emptyList()

        return try {
            val chars = word.lowercase().map { it.toString() }
            val indices = LongArray(chars.size) { i ->
                (graphemeToIndex[chars[i]] ?: 0).toLong()
            }

            OnnxTensor.createTensor(
                ortEnv,
                LongBuffer.wrap(indices),
                longArrayOf(1, indices.size.toLong())
            ).use { inputTensor ->
                val inputName = s.inputNames.firstOrNull() ?: return emptyList()
                s.run(mapOf(inputName to inputTensor)).use { result ->
                    val outputTensor = result.get(0)?.value as? OnnxTensor
                        ?: return emptyList()
                    val buf = outputTensor.longBuffer
                    val phonemeIndices = LongArray(buf.remaining())
                    buf.get(phonemeIndices)

                    buildList {
                        for (idx in phonemeIndices) {
                            if (idx <= 0L || idx >= phonemeVocab.size.toLong()) continue
                            val ph = indexToPhoneme[idx.toInt()]
                            if (!ph.isNullOrEmpty()) add(ph)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    /**
     * Closes the ONNX session and releases its resources.
     * Does not close the shared [OrtEnvironment].
     */
    fun close() {
        session?.close()
        session = null
    }
}
