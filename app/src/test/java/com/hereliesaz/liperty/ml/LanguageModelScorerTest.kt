package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tests for [LanguageModelScorer] and the supplied no-op implementation.
 *
 * The interface itself is intentionally tiny — `score(words)` returns the
 * log10 probability of the word sequence under whatever LM the
 * implementation wraps. Real implementations (e.g. [KenLmScorer]) wrap a
 * native KenLM via JNI. The interface exists so [SubwordCtcBeamDecoder]
 * can take an LM without coupling to a particular native library, and so
 * tests can inject deterministic fakes.
 */
class LanguageModelScorerTest {

    @Test
    fun noopScorerAlwaysReturnsZero() {
        val s = NoopLanguageModelScorer()
        assertEquals(0f, s.score(emptyList()), 0f)
        assertEquals(0f, s.score(listOf("the", "cat")), 0f)
        s.close()
    }

    @Test
    fun fakeScorerAddsToBeamRankingWithoutChangingApi() {
        // A scorer that boosts the phrase "the cat sat on the mat" but
        // penalizes anything else. Used by SubwordCtcBeamDecoderRescoringTest
        // to verify rescoring promotes the LM-preferred beam.
        val good = "the cat sat on the mat"
        val s = object : LanguageModelScorer {
            override fun score(words: List<String>): Float =
                if (words.joinToString(" ") == good) -1f else -50f
            override fun close() {}
        }
        assertEquals(-1f, s.score(listOf("the", "cat", "sat", "on", "the", "mat")), 0f)
        assertEquals(-50f, s.score(listOf("the", "cat", "math", "on", "the", "mat")), 0f)
    }
}
