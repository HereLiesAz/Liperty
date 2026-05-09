package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tests for [SubwordCTCDecoder] — greedy CTC decoder over a SentencePiece-style
 * subword vocabulary (the format used by Auto-AVSR's `unigram5000_units.txt`).
 *
 * Conventions:
 *   - Index 0 = blank.
 *   - Tokens beginning with `▁` (U+2581) are word-initial; the `▁` is replaced
 *     with a space at detokenization time.
 *   - Tokens not beginning with `▁` are subword continuations and join without
 *     a separator.
 *
 * The argmax-collapse-blank logic is identical to standard CTC. Detokenization
 * is the new piece relative to the existing phoneme [GreedyDecoder].
 */
class SubwordCTCDecoderTest {

    private val vocab = listOf(
        "_",        // 0  blank
        "▁the", // 1  ▁the
        "▁cat", // 2  ▁cat
        "ten",       // 3
        "ed",        // 4
        "▁sat", // 5  ▁sat
    )

    private fun oneHotProbs(indices: IntArray, vocabSize: Int): Array<FloatArray> =
        Array(indices.size) { t ->
            FloatArray(vocabSize).also { it[indices[t]] = 1.0f }
        }

    @Test
    fun decodesIndependentTokensIntoSpaceJoinedWords() {
        val decoder = SubwordCTCDecoder(vocab)
        val probs = oneHotProbs(intArrayOf(1, 2, 5), vocab.size)
        // ▁the, ▁cat, ▁sat → "the cat sat"
        assertEquals("the cat sat", decoder.decode(probs))
    }

    @Test
    fun joinsSubwordContinuationsWithoutSpace() {
        val decoder = SubwordCTCDecoder(vocab)
        // ▁the, ten, ed → "theten" + "ed" → "thetened"
        // (illustrative only; the real model would never predict this nonsense)
        val probs = oneHotProbs(intArrayOf(1, 3, 4), vocab.size)
        assertEquals("thetened", decoder.decode(probs))
    }

    @Test
    fun collapsesConsecutiveDuplicateIndices() {
        val decoder = SubwordCTCDecoder(vocab)
        // ▁the, ▁the (no blank between) → single ▁the
        val probs = oneHotProbs(intArrayOf(1, 1, 2), vocab.size)
        assertEquals("the cat", decoder.decode(probs))
    }

    @Test
    fun keepsRepeatedTokensSeparatedByBlank() {
        val decoder = SubwordCTCDecoder(vocab)
        // ▁the, blank, ▁the → two emissions
        val probs = oneHotProbs(intArrayOf(1, 0, 1), vocab.size)
        assertEquals("the the", decoder.decode(probs))
    }

    @Test
    fun stripsBlankFramesFromOutput() {
        val decoder = SubwordCTCDecoder(vocab)
        val probs = oneHotProbs(intArrayOf(0, 1, 0, 0, 5, 0), vocab.size)
        assertEquals("the sat", decoder.decode(probs))
    }

    @Test
    fun returnsEmptyStringForAllBlankInput() {
        val decoder = SubwordCTCDecoder(vocab)
        val probs = oneHotProbs(intArrayOf(0, 0, 0), vocab.size)
        assertEquals("", decoder.decode(probs))
    }

    @Test
    fun ignoresIndicesOutOfVocabRange() {
        val decoder = SubwordCTCDecoder(vocab)
        val probs = Array(2) { FloatArray(vocab.size + 5) }
        probs[0][1] = 1.0f      // ▁the (in range)
        probs[1][vocab.size + 2] = 1.0f  // out of range — ignored
        assertEquals("the", decoder.decode(probs))
    }

    @Test
    fun pickArgmaxAcrossTimesteps() {
        val decoder = SubwordCTCDecoder(vocab)
        val probs = Array(2) { FloatArray(vocab.size) }
        // Soft probabilities, argmax over each timestep
        probs[0][1] = 0.7f; probs[0][2] = 0.2f; probs[0][0] = 0.1f
        probs[1][2] = 0.6f; probs[1][1] = 0.3f; probs[1][0] = 0.1f
        assertEquals("the cat", decoder.decode(probs))
    }
}
