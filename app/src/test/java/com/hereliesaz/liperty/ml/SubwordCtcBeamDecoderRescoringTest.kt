package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tests for the OPTIONAL n-best rescoring path in [SubwordCtcBeamDecoder].
 *
 * The decoder keeps its top-K beams (where K = beamWidth) sorted by CTC
 * score after the time loop. When an [LanguageModelScorer] is supplied,
 * each candidate's word sequence is scored with the LM, and the final
 * winner is `argmax_b (CTC_score(b) + lmWeight * LM_score(b))`.
 *
 * Backward compat: omitting the LM (or supplying [NoopLanguageModelScorer])
 * must yield identical output to the un-rescored decoder.
 */
class SubwordCtcBeamDecoderRescoringTest {

    private val vocab = listOf(
        "_",        // 0 blank
        "▁the",     // 1
        "▁cat",     // 2
        "▁math",    // 3
        "▁mat",     // 4
        "▁sat",     // 5
        "▁on",      // 6
    )

    /** Build a (T, V) probability matrix where each row is a mixture over two
     *  vocab indices (the "intended" and "alternate" token). Forces the
     *  greedy/no-LM path to pick the intended, and the LM-rescored path to
     *  pick whichever it prefers if the per-row gap is small enough that the
     *  alternate beam survives to the rescoring step. */
    private fun mixedProbs(rows: List<Pair<Int, Int>>, gap: Float = 0.02f): Array<FloatArray> =
        Array(rows.size) { t ->
            FloatArray(vocab.size).also {
                val (intended, alternate) = rows[t]
                it[intended] = 0.5f + gap
                it[alternate] = 0.5f - gap
            }
        }

    @Test
    fun nullLmYieldsIdenticalOutputToBackCompatPath() {
        val baseline = SubwordCtcBeamDecoder(vocab, beamWidth = 8)
        val withNoop = SubwordCtcBeamDecoder(vocab, beamWidth = 8,
            lmScorer = NoopLanguageModelScorer(), lmWeight = 5f)

        val probs = mixedProbs(listOf(1 to 2, 5 to 4))
        assertEquals(baseline.decode(probs), withNoop.decode(probs))
    }

    @Test
    fun rescoringPromotesLmPreferredBeam() {
        // Intended ("the cat") wins greedily but LM strongly prefers "the sat"
        // (contrived; the test is about ranking, not linguistic plausibility).
        val probs = mixedProbs(listOf(1 to 1, 2 to 5))  // → "the cat" or "the sat"

        val baseline = SubwordCtcBeamDecoder(vocab, beamWidth = 8)
        assertEquals("the cat", baseline.decode(probs))

        val lm = object : LanguageModelScorer {
            override fun score(words: List<String>): Float = when (words) {
                listOf("the", "sat") -> -1f
                listOf("the", "cat") -> -100f
                else -> -200f
            }
            override fun close() {}
        }
        val rescored = SubwordCtcBeamDecoder(vocab, beamWidth = 8,
            lmScorer = lm, lmWeight = 1f)
        assertEquals("the sat", rescored.decode(probs))
    }

    @Test
    fun zeroLmWeightDisablesRescoringEvenWhenScorerIsPresent() {
        // A scorer that loves "the sat" — but lmWeight=0 -> it never wins.
        val probs = mixedProbs(listOf(1 to 1, 2 to 5))
        val lm = object : LanguageModelScorer {
            override fun score(words: List<String>): Float =
                if (words == listOf("the", "sat")) 1000f else -1000f
            override fun close() {}
        }
        val decoder = SubwordCtcBeamDecoder(vocab, beamWidth = 8,
            lmScorer = lm, lmWeight = 0f)
        assertEquals("the cat", decoder.decode(probs))
    }

    @Test
    fun lmReceivesWordsSplitOnBoundaryMarker() {
        // Need multiple surviving beams so the LM scorer is actually invoked
        // — Kotlin's maxByOrNull short-circuits on a singleton, which is the
        // correct optimization (LM can't change a 1-way race) but means the
        // test must construct genuine competition between candidates.
        // Two-row mix with DIFFERENT intended/alternate indices → 4 paths.
        val probs = mixedProbs(listOf(1 to 5, 2 to 4))
        val seen = ArrayList<List<String>>()
        val lm = object : LanguageModelScorer {
            override fun score(words: List<String>): Float {
                seen.add(words.toList())
                return 0f
            }
            override fun close() {}
        }
        SubwordCtcBeamDecoder(vocab, beamWidth = 8,
            lmScorer = lm, lmWeight = 1f).decode(probs)
        // The decoder splits raw beam text "▁the▁cat" into ["the","cat"]
        // before scoring — that's the LM's expected input shape.
        assertEquals(true, seen.any { it == listOf("the", "cat") })
        assertEquals(true, seen.all { ws -> ws.all { '▁' !in it } })
    }
}
