package com.hereliesaz.liperty.ml

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for [LlmTextCleaner] — wraps a small on-device LLM (e.g. Gemma-2B-it
 * or Phi-3-mini, served via MediaPipe Tasks GenAI's `LlmInference`) to clean
 * up the noisy CTC output from the visual-only Auto-AVSR encoder.
 *
 * The class is designed for testability: the constructor takes a `generate`
 * suspend lambda that produces a model response for a given prompt. Real
 * deployment passes `LlmInference::generateResponse` (or its async variant);
 * tests pass a fake that returns deterministic strings.
 */
class LlmTextCleanerTest {

    private fun cleanerReturning(response: String): LlmTextCleaner =
        LlmTextCleaner { _ -> response }

    private fun captureCleaner(captured: MutableList<String>): LlmTextCleaner =
        LlmTextCleaner { prompt -> captured.add(prompt); "ignored" }

    @Test
    fun emptyInputReturnsEmpty() = runBlocking {
        val cleaner = cleanerReturning("anything")
        assertEquals("", cleaner.clean(""))
    }

    @Test
    fun whitespaceOnlyInputReturnsTrimmedEmpty() = runBlocking {
        val cleaner = cleanerReturning("anything")
        assertEquals("", cleaner.clean("   \n\t "))
    }

    @Test
    fun promptContainsTheInputText() = runBlocking {
        val captured = mutableListOf<String>()
        val cleaner = captureCleaner(captured)
        cleaner.clean("th cat sa on the mt")
        assertEquals(1, captured.size)
        assertTrue("Prompt should contain raw input. Got: ${captured[0]}",
            captured[0].contains("th cat sa on the mt"))
    }

    @Test
    fun promptHasInstructionThatRequestsOnlyCorrectedText() = runBlocking {
        val captured = mutableListOf<String>()
        val cleaner = captureCleaner(captured)
        cleaner.clean("hello")
        // Prompt must steer the model toward outputting ONLY the correction —
        // small instruction-tuned models love to add chatter otherwise.
        val p = captured[0].lowercase()
        assertTrue("Prompt should request output-only behavior. Got: $p",
            p.contains("only") || p.contains("nothing else") || p.contains("don't"))
    }

    @Test
    fun simpleResponseIsReturnedAsIs() = runBlocking {
        val cleaner = cleanerReturning("the cat sat on the mat")
        assertEquals("the cat sat on the mat", cleaner.clean("th cat sa on the mt"))
    }

    @Test
    fun stripsLeadingAndTrailingWhitespace() = runBlocking {
        val cleaner = cleanerReturning("   the cat sat on the mat\n\n  ")
        assertEquals("the cat sat on the mat", cleaner.clean("th cat sa"))
    }

    @Test
    fun stripsLeadingChatterPrefix() = runBlocking {
        // Small models love to say "Sure, here's the corrected text:" first.
        val cleaner = cleanerReturning("Sure, here's the corrected text: the cat sat on the mat")
        assertEquals("the cat sat on the mat", cleaner.clean("th cat sa"))
    }

    @Test
    fun stripsCorrectedColonPrefix() = runBlocking {
        val cleaner = cleanerReturning("Corrected: the cat sat on the mat")
        assertEquals("the cat sat on the mat", cleaner.clean("th cat sa"))
    }

    @Test
    fun stripsMarkdownQuotesAroundResponse() = runBlocking {
        val cleaner = cleanerReturning("\"the cat sat on the mat\"")
        assertEquals("the cat sat on the mat", cleaner.clean("th cat sa"))
    }

    @Test
    fun stripsBackticksAroundResponse() = runBlocking {
        val cleaner = cleanerReturning("`the cat sat on the mat`")
        assertEquals("the cat sat on the mat", cleaner.clean("th cat sa"))
    }

    @Test
    fun multilineResponseTakesFirstNonEmptyLine() = runBlocking {
        val cleaner = cleanerReturning("\nthe cat sat on the mat\n\nLet me know if you'd like more!\n")
        assertEquals("the cat sat on the mat", cleaner.clean("th cat sa"))
    }

    @Test
    fun emptyResponseFallsBackToRawInput() = runBlocking {
        val cleaner = cleanerReturning("")
        assertEquals("th cat sa on the mt", cleaner.clean("th cat sa on the mt"))
    }

    @Test
    fun whitespaceOnlyResponseFallsBackToRawInput() = runBlocking {
        val cleaner = cleanerReturning("   \n  \n   ")
        assertEquals("th cat sa", cleaner.clean("th cat sa"))
    }

    @Test
    fun responseFromCallableThrowingFallsBackToRawInput() = runBlocking {
        val cleaner = LlmTextCleaner { _ -> throw RuntimeException("model timeout") }
        assertEquals("th cat sa", cleaner.clean("th cat sa"))
    }

    @Test
    fun cleanIsNoOpWhenInputIsAlreadyClean() = runBlocking {
        // We can't really know, but if the LLM echoes back unchanged, that's
        // the right outcome.
        val cleaner = cleanerReturning("the cat sat on the mat")
        assertEquals("the cat sat on the mat", cleaner.clean("the cat sat on the mat"))
    }
}
