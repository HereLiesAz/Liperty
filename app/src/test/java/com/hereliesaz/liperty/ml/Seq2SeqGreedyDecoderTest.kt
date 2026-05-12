package com.hereliesaz.liperty.ml

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for [Seq2SeqGreedyDecoder].
 *
 * The decoder owns the autoregressive loop and an EOS check. The step
 * function (i.e. the decoder ONNX call) is injected so the loop can be
 * tested in pure JVM with a deterministic dummy. Encoder features are
 * not the decoder's concern — they're captured in the closure of the
 * step function by the orchestrator.
 *
 * Contract:
 *   - decode(step) returns the full token sequence including the
 *     initial BOS and any terminating EOS.
 *   - On each iteration the loop calls step(currentTokens) and takes
 *     argmax of the returned (vocab,) FloatArray to pick the next token.
 *   - On EOS, return immediately (don't emit beyond).
 *   - On hitting maxSteps without EOS, return what we have so far.
 */
class Seq2SeqGreedyDecoderTest {

    /** Tiny vocab: 0=bos, 1=pad, 2=eos, 3=unk, then 4..9 = "letters" A..F. */
    private val bos = 0
    private val eos = 2
    private val vocabSize = 10

    /** Build a one-hot logits vector that argmaxes to [picked]. */
    private fun oneHot(picked: Int): FloatArray =
        FloatArray(vocabSize).also { it[picked] = 1f }

    @Test
    fun emitsBosAndStopsAtEos() {
        val decoder = Seq2SeqGreedyDecoder(bosId = bos, eosId = eos, maxSteps = 20)
        // Step plays out: bos -> A(4) -> B(5) -> eos
        val plan = intArrayOf(4, 5, eos)
        var step = 0
        val tokens = decoder.decode { prev ->
            // Length check: prev grows by 1 each call, starting at [bos]
            assertEquals(step + 1, prev.size)
            assertEquals(bos, prev[0])
            oneHot(plan[step++])
        }
        // Expected output: [bos, 4, 5, eos]
        assertArrayEquals(intArrayOf(bos, 4, 5, eos), tokens)
        assertEquals(3, step)   // step fn called exactly 3 times
    }

    @Test
    fun stopsAtMaxStepsWithoutEosWhenLoopBudgetExhausted() {
        val decoder = Seq2SeqGreedyDecoder(bosId = bos, eosId = eos, maxSteps = 4)
        // Step always picks A — would loop forever without the budget.
        var calls = 0
        val tokens = decoder.decode { _ -> calls++; oneHot(4) }
        // Expected: bos + 4 emitted tokens (loop budget) = 5 entries.
        assertEquals(5, tokens.size)
        assertEquals(bos, tokens[0])
        for (i in 1..4) assertEquals(4, tokens[i])
        assertEquals(4, calls)
    }

    @Test
    fun immediateEosProducesBosEosOnly() {
        val decoder = Seq2SeqGreedyDecoder(bosId = bos, eosId = eos, maxSteps = 20)
        var calls = 0
        val tokens = decoder.decode { _ -> calls++; oneHot(eos) }
        // First (and only) decoder call returns eos -> output is [bos, eos].
        assertArrayEquals(intArrayOf(bos, eos), tokens)
        assertEquals(1, calls)
    }

    @Test
    fun passesGrowingTokenHistoryToStep() {
        // The step function should see the full history of emitted tokens
        // each call, ending at the most recent (NOT just the latest).
        val decoder = Seq2SeqGreedyDecoder(bosId = bos, eosId = eos, maxSteps = 3)
        val seen = ArrayList<IntArray>()
        decoder.decode { prev ->
            seen.add(prev.copyOf())
            oneHot(4)
        }
        assertEquals(3, seen.size)
        assertArrayEquals(intArrayOf(bos), seen[0])
        assertArrayEquals(intArrayOf(bos, 4), seen[1])
        assertArrayEquals(intArrayOf(bos, 4, 4), seen[2])
    }

    @Test
    fun usesArgmaxNotProbabilityForTokenSelection() {
        // Logits don't need to be normalized. The decoder must pick the
        // highest-scoring index regardless of sign/scale.
        val decoder = Seq2SeqGreedyDecoder(bosId = bos, eosId = eos, maxSteps = 1)
        val raw = FloatArray(vocabSize) { i -> if (i == 7) 100f else -1000f }
        val tokens = decoder.decode { _ -> raw }
        // Should pick token 7 then stop at maxSteps=1.
        assertArrayEquals(intArrayOf(bos, 7), tokens)
    }

    @Test
    fun maxStepsZeroReturnsBosOnly() {
        // Edge case: no autoregressive steps requested -> just return [bos].
        val decoder = Seq2SeqGreedyDecoder(bosId = bos, eosId = eos, maxSteps = 0)
        var calls = 0
        val tokens = decoder.decode { _ -> calls++; oneHot(4) }
        assertArrayEquals(intArrayOf(bos), tokens)
        assertEquals(0, calls)
    }
}
