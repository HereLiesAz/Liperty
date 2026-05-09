package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for [SubwordCtcBeamDecoder] — CTC beam search over a SentencePiece
 * vocabulary. Same word-boundary detokenization rules as
 * [SubwordCTCDecoder] (the greedy variant), but explores a beam of paths
 * instead of just argmax.
 *
 * The beam search adds value over greedy when:
 *   - Two tokens have similar per-step probability and one is more likely
 *     in context (CTC's path-collapsing makes greedy myopic).
 *   - Repeats need to be disambiguated via blank insertion.
 *
 * For one-hot inputs (per-timestep argmax dominates), the beam result must
 * match the greedy result -- this is the back-compat property tested first.
 */
class SubwordCtcBeamDecoderTest {

    private val vocab = listOf(
        "_",       // 0  blank
        "▁the",    // 1
        "▁cat",    // 2
        "ten",     // 3
        "ed",      // 4
        "▁sat",    // 5
    )

    private fun oneHot(indices: IntArray, V: Int = vocab.size): Array<FloatArray> =
        Array(indices.size) { t -> FloatArray(V).also { it[indices[t]] = 1.0f } }

    @Test
    fun matchesGreedyOnOneHotInput() {
        val decoder = SubwordCtcBeamDecoder(vocab, beamWidth = 4)
        val probs = oneHot(intArrayOf(1, 2, 5))
        assertEquals("the cat sat", decoder.decode(probs))
    }

    @Test
    fun joinsSubwordContinuationsWithoutSpace() {
        val decoder = SubwordCtcBeamDecoder(vocab, beamWidth = 4)
        val probs = oneHot(intArrayOf(1, 3, 4))   // ▁the, ten, ed
        assertEquals("thetened", decoder.decode(probs))
    }

    @Test
    fun collapsesCtcRepeatsWithoutBlank() {
        val decoder = SubwordCtcBeamDecoder(vocab, beamWidth = 4)
        val probs = oneHot(intArrayOf(1, 1, 2))   // ▁the, ▁the, ▁cat → "the cat"
        assertEquals("the cat", decoder.decode(probs))
    }

    @Test
    fun keepsRepeatedTokensSeparatedByBlank() {
        val decoder = SubwordCtcBeamDecoder(vocab, beamWidth = 4)
        val probs = oneHot(intArrayOf(1, 0, 1))   // ▁the, blank, ▁the → "the the"
        assertEquals("the the", decoder.decode(probs))
    }

    @Test
    fun stripsBlankFrames() {
        val decoder = SubwordCtcBeamDecoder(vocab, beamWidth = 4)
        val probs = oneHot(intArrayOf(0, 1, 0, 0, 5, 0))
        assertEquals("the sat", decoder.decode(probs))
    }

    @Test
    fun emptyInputReturnsEmptyString() {
        val decoder = SubwordCtcBeamDecoder(vocab, beamWidth = 4)
        assertEquals("", decoder.decode(emptyArray()))
    }

    @Test
    fun allBlankInputReturnsEmptyString() {
        val decoder = SubwordCtcBeamDecoder(vocab, beamWidth = 4)
        val probs = oneHot(intArrayOf(0, 0, 0, 0))
        assertEquals("", decoder.decode(probs))
    }

    @Test
    fun pickHigherTotalScoreWhenGreedyDisagrees() {
        // Construct a case where the locally-best per-step token (greedy) leads
        // to a worse total path than a slightly-worse-per-step alternative.
        //
        // Step 0: ▁the (0.6), ▁cat (0.4)
        // Step 1: ▁the (0.5), ▁cat (0.5)
        // Step 2: ▁sat (0.7), ▁the (0.3)
        //
        // Greedy: ▁the ▁the ▁sat -> CTC collapse -> "the sat"
        // Beam should ALSO get "the sat" -- this case still has a clear winner.
        // We just verify the beam returns something coherent (not garbled).
        val probs = arrayOf(
            FloatArray(vocab.size).apply { this[1] = 0.6f; this[2] = 0.4f },
            FloatArray(vocab.size).apply { this[1] = 0.5f; this[2] = 0.5f },
            FloatArray(vocab.size).apply { this[5] = 0.7f; this[1] = 0.3f },
        )
        val result = SubwordCtcBeamDecoder(vocab, beamWidth = 4).decode(probs)
        // Either "the sat" or "the cat sat" — both reasonable; both valid.
        assertTrue("Got: '$result'", result == "the sat" || result == "the cat sat")
    }

    @Test
    fun beamWidthOneEqualsGreedy() {
        // With beam width 1, we degenerate to greedy. Spot-check parity on a
        // sequence where path-collapsing matters.
        val beam1 = SubwordCtcBeamDecoder(vocab, beamWidth = 1)
        val greedy = SubwordCTCDecoder(vocab)
        val probs = oneHot(intArrayOf(1, 1, 0, 1, 2, 5))
        assertEquals(greedy.decode(probs), beam1.decode(probs))
    }

    @Test
    fun ignoresIndicesOutOfVocabRange() {
        val decoder = SubwordCtcBeamDecoder(vocab, beamWidth = 4)
        val probs = Array(2) { FloatArray(vocab.size + 5) }
        probs[0][1] = 1.0f                   // ▁the
        probs[1][vocab.size + 2] = 1.0f      // out of range
        // Out-of-range token gets ignored; only ▁the contributes.
        assertEquals("the", decoder.decode(probs))
    }
}
