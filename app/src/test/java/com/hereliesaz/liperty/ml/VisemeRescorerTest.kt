package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tests for [VisemeRescorer] — the "Chaplin's-second-AI" layer specific
 * to visual ASR. Given a CTC-decoded sentence, expands each word to
 * viseme-equivalent candidates and uses an external [LanguageModelScorer]
 * to pick the most English-plausible substitution.
 *
 * Key behaviors:
 *   - With a no-op LM, the original sentence is returned (no signal to
 *     pick anyone else; the rescorer must default to input on ties).
 *   - With a discriminating LM, the rescorer picks the highest-scoring
 *     viseme-equivalent in context.
 *   - Per-word substitution is independent — the rescorer commits to
 *     one winner per position (left-to-right greedy is enough for short
 *     sentences; beam can come later if WER shows tail errors).
 *   - Empty input passes through.
 *   - Unknown words pass through unchanged.
 */
class VisemeRescorerTest {

    /** Tiny index: "tasty" and "nasty" share visemes; "bird" is alone;
     *  "flew" / "blew" / "drew" share visemes. */
    private val index = VisemeIndex(mapOf(
        "V4 V1 V4 V4 V1" to listOf("NASTY", "TASTY"),
        "V2 V6 V4" to listOf("BIRD"),
        "V2 V4 V9" to listOf("BLEW", "DREW", "FLEW"),     // contrived, for tests
    ))

    @Test
    fun returnsInputWhenLmIsNoop() {
        val r = VisemeRescorer(index, NoopLanguageModelScorer())
        assertEquals("bird flew tasty", r.rescore("bird flew tasty"))
    }

    @Test
    fun picksHigherLmScoringVisemeEquivalent() {
        // LM prefers "bird flew nasty" over "bird flew tasty".
        val lm = object : LanguageModelScorer {
            override fun score(words: List<String>): Float {
                val s = words.joinToString(" ")
                return when {
                    s.endsWith("nasty") -> -2f
                    s.endsWith("tasty") -> -10f
                    else -> -100f
                }
            }
            override fun close() {}
        }
        val r = VisemeRescorer(index, lm)
        assertEquals("bird flew nasty", r.rescore("bird flew tasty"))
    }

    @Test
    fun preservesUnknownWords() {
        // "elephant" isn't in our tiny index → must pass through unchanged.
        val r = VisemeRescorer(index, NoopLanguageModelScorer())
        assertEquals("elephant bird", r.rescore("elephant bird"))
    }

    @Test
    fun emptyInputReturnsEmpty() {
        val r = VisemeRescorer(index, NoopLanguageModelScorer())
        assertEquals("", r.rescore(""))
        assertEquals("", r.rescore("   "))
    }

    @Test
    fun preservesCasingOfPassThroughWords() {
        // The rescorer's output for words it doesn't substitute should
        // match the input. (Substituted words come from the index, which
        // is uppercase, then lowercased on output.)
        val r = VisemeRescorer(index, NoopLanguageModelScorer())
        assertEquals("elephant bird", r.rescore("elephant bird"))
    }

    @Test
    fun substitutionsAreLowercaseInOutput() {
        // Index entries are uppercase (cmudict convention). The rescorer
        // lowercases substituted words so the output reads naturally.
        val lm = object : LanguageModelScorer {
            override fun score(words: List<String>): Float =
                if ("nasty" in words.joinToString(" ").lowercase()) -2f else -10f
            override fun close() {}
        }
        val r = VisemeRescorer(index, lm)
        val out = r.rescore("bird flew tasty")
        assertEquals("bird flew nasty", out)
    }

    @Test
    fun greedyConsidersContextNotJustLocalScore() {
        // The LM scores full sentences, so substitutions in earlier
        // positions affect later choices. Specifically: if "flew" makes
        // "tasty" more likely afterward, and "blew" makes "nasty" more
        // likely, the rescorer chooses the higher GLOBAL score.
        val lm = object : LanguageModelScorer {
            override fun score(words: List<String>): Float = when (words.joinToString(" ")) {
                "bird blew nasty" -> -1f       // best overall
                "bird flew tasty" -> -2f
                "bird flew nasty" -> -3f
                "bird blew tasty" -> -4f
                else -> -100f
            }
            override fun close() {}
        }
        val r = VisemeRescorer(index, lm)
        assertEquals("bird blew nasty", r.rescore("bird flew tasty"))
    }
}
