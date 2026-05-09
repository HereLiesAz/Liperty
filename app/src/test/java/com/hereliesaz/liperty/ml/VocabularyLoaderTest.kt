package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tests for [VocabularyLoader] — parses Auto-AVSR / ESPnet-style token files
 * such as `unigram5000_units.txt`, where each line is `<token><whitespace><index>`
 * with 1-based indices.
 *
 * Two responsibilities:
 *   - [VocabularyLoader.parseTokenList] strips the index column and returns the
 *     tokens in file order. Pure function; the file IO is the caller's job.
 *   - [VocabularyLoader.withBlankAtZero] prepends a blank token so the resulting
 *     list maps directly to a CTC head whose index 0 is the blank.
 */
class VocabularyLoaderTest {

    @Test
    fun parsesTokensInFileOrderStrippingIndex() {
        val text = """
            <unk> 1
            ' 2
            0 3
        """.trimIndent()
        val tokens = VocabularyLoader.parseTokenList(text)
        assertEquals(listOf("<unk>", "'", "0"), tokens)
    }

    @Test
    fun preservesSentencepieceWordBoundaryMarker() {
        val text = """
            ▁the 1
            cat 2
            ▁sat 3
        """.trimIndent()
        val tokens = VocabularyLoader.parseTokenList(text)
        assertEquals(listOf("▁the", "cat", "▁sat"), tokens)
    }

    @Test
    fun tolerantOfBlankLinesAndTrailingWhitespace() {
        val text = "\n  abc 1\n\n  def 2  \n\n"
        val tokens = VocabularyLoader.parseTokenList(text)
        assertEquals(listOf("abc", "def"), tokens)
    }

    @Test
    fun acceptsTabSeparatedFormat() {
        // Some SentencePiece dumps use tab between token and index.
        val text = "abc\t1\ndef\t2"
        val tokens = VocabularyLoader.parseTokenList(text)
        assertEquals(listOf("abc", "def"), tokens)
    }

    @Test
    fun acceptsTokenLineWithoutIndexColumn() {
        // Some preprocess pipelines emit just one token per line, no index.
        val text = "abc\ndef\nghi"
        val tokens = VocabularyLoader.parseTokenList(text)
        assertEquals(listOf("abc", "def", "ghi"), tokens)
    }

    @Test
    fun emptyInputReturnsEmptyList() {
        assertEquals(emptyList<String>(), VocabularyLoader.parseTokenList(""))
        assertEquals(emptyList<String>(), VocabularyLoader.parseTokenList("   \n\n  "))
    }

    @Test
    fun withBlankAtZeroPrependsBlankToken() {
        val tokens = listOf("a", "b", "c")
        val withBlank = VocabularyLoader.withBlankAtZero(tokens, blank = "_")
        assertEquals(listOf("_", "a", "b", "c"), withBlank)
    }

    @Test
    fun withBlankAtZeroDoesNotDuplicateExistingBlank() {
        // If the caller already provides a vocab whose first entry is the blank,
        // don't prepend a second one.
        val tokens = listOf("_", "a", "b")
        val withBlank = VocabularyLoader.withBlankAtZero(tokens, blank = "_")
        assertEquals(listOf("_", "a", "b"), withBlank)
    }
}
