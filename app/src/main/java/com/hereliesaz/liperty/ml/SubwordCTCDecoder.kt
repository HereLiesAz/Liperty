package com.hereliesaz.liperty.ml

/**
 * Greedy CTC decoder over a SentencePiece-style subword vocabulary — the
 * format used by the Auto-AVSR / VSR-ML LRS3 checkpoint Liperty exports via
 * `tools/export_autoavsr_to_onnx.ipynb`.
 *
 * Vocabulary convention:
 *   - Index 0 is the CTC blank.
 *   - Tokens beginning with `▁` (U+2581) mark word boundaries — at
 *     detokenization the leading `▁` becomes a space.
 *   - Tokens not beginning with `▁` are subword continuations and join
 *     directly to the previous emission.
 *
 * Decoding is the standard CTC-greedy collapse:
 *   1. Per timestep, take argmax.
 *   2. Skip indices equal to the previous emitted index (collapse repeats).
 *   3. Skip the blank index entirely (and reset the previous-index tracker
 *      so a real repeat across a blank gets emitted twice).
 *   4. Skip indices outside the vocabulary range defensively.
 */
class SubwordCTCDecoder(
    private val vocabulary: List<String>,
    private val blankIndex: Int = 0,
) {
    /**
     * @param probabilities `(T, V)` softmax probabilities — caller may also pass
     *   raw logits; only argmax is used so the absolute scale doesn't matter.
     * @return the decoded text, with subword units joined and `▁` markers
     *   replaced by spaces. Leading whitespace is stripped.
     */
    fun decode(probabilities: Array<FloatArray>): String {
        val emitted = StringBuilder()
        var prevIndex = -1
        for (timeStep in probabilities) {
            val maxIndex = argmax(timeStep)
            if (maxIndex < 0) {
                continue
            }
            if (maxIndex == blankIndex) {
                prevIndex = -1
                continue
            }
            if (maxIndex == prevIndex) {
                continue
            }
            if (maxIndex < vocabulary.size) {
                emitted.append(vocabulary[maxIndex])
            }
            prevIndex = maxIndex
        }
        return detokenize(emitted.toString())
    }

    private fun argmax(timeStep: FloatArray): Int {
        if (timeStep.isEmpty()) return -1
        var best = 0
        var bestVal = timeStep[0]
        for (i in 1 until timeStep.size) {
            if (timeStep[i] > bestVal) {
                bestVal = timeStep[i]
                best = i
            }
        }
        return best
    }

    private fun detokenize(joined: String): String =
        joined.replace(WORD_BOUNDARY, ' ').trimStart()

    companion object {
        /** SentencePiece word-boundary marker. */
        const val WORD_BOUNDARY: Char = '▁'
    }
}
