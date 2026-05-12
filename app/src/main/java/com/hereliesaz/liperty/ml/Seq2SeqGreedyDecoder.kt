package com.hereliesaz.liperty.ml

/**
 * Greedy autoregressive decoder for the AV-HuBERT seq2seq path. Owns the
 * loop (start at BOS, append argmax each step, stop on EOS or budget).
 * The actual decoder ONNX call is injected as a step lambda so this
 * class is pure JVM and easy to unit-test.
 *
 * Beam search is a separate class. Greedy is the right default for a
 * first integration — every extra beam multiplies the per-utterance
 * ONNX cost — and the V3 backend is gated on beating V2 with the
 * simplest possible decode anyway.
 *
 * @param bosId beginning-of-sequence token (fairseq dict index 0 for AV-HuBERT)
 * @param eosId end-of-sequence token (fairseq dict index 2 for AV-HuBERT)
 * @param maxSteps how many tokens to emit beyond BOS before bailing out.
 *   Set to a sane utterance ceiling; 50 BPE tokens is roughly a 10-15-word
 *   sentence, which is well past Liperty's expected utterance length.
 */
class Seq2SeqGreedyDecoder(
    private val bosId: Int,
    private val eosId: Int,
    private val maxSteps: Int = 50,
) {

    /**
     * Run the autoregressive loop.
     *
     * @param step Called once per autoregressive position with the full
     *   token history so far (always starting with [bosId]). Must return
     *   a (vocabSize,) `FloatArray` of logits for the NEXT position; the
     *   loop takes its argmax.
     * @return the emitted token sequence, including the initial BOS and
     *   any terminating EOS.
     */
    fun decode(step: (IntArray) -> FloatArray): IntArray {
        var tokens = intArrayOf(bosId)
        for (i in 0 until maxSteps) {
            val logits = step(tokens)
            val next = argmax(logits)
            tokens += next
            if (next == eosId) return tokens
        }
        return tokens
    }

    private fun argmax(logits: FloatArray): Int {
        var best = 0
        var bestV = logits[0]
        for (i in 1 until logits.size) {
            if (logits[i] > bestV) {
                bestV = logits[i]
                best = i
            }
        }
        return best
    }
}
