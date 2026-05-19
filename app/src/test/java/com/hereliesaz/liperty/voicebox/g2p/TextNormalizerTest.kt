package com.hereliesaz.liperty.voicebox.g2p

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pure JUnit 4 tests for [TextNormalizer].
 * No Robolectric or Android Context is required.
 */
class TextNormalizerTest {

    // ── numberToWords ─────────────────────────────────────────────────────

    @Test
    fun `zero returns zero`() {
        assertEquals("zero", TextNormalizer.numberToWords(0L))
    }

    @Test
    fun `one returns one`() {
        assertEquals("one", TextNormalizer.numberToWords(1L))
    }

    @Test
    fun `42 returns forty-two`() {
        assertEquals("forty-two", TextNormalizer.numberToWords(42L))
    }

    @Test
    fun `100 returns one hundred`() {
        assertEquals("one hundred", TextNormalizer.numberToWords(100L))
    }

    @Test
    fun `1000 returns one thousand`() {
        assertEquals("one thousand", TextNormalizer.numberToWords(1_000L))
    }

    @Test
    fun `999999 returns correct words`() {
        val result = TextNormalizer.numberToWords(999_999L)
        assertTrue("should contain 'nine hundred'", result.contains("nine hundred"))
        assertTrue("should contain 'thousand'", result.contains("thousand"))
        assertTrue("should contain 'ninety-nine'", result.contains("ninety-nine"))
    }

    @Test
    fun `999999999 converts correctly`() {
        val result = TextNormalizer.numberToWords(999_999_999L)
        assertTrue(result.contains("million"))
        assertTrue(result.contains("thousand"))
    }

    @Test
    fun `out of range returns string representation`() {
        assertEquals("1000000000", TextNormalizer.numberToWords(1_000_000_000L))
    }

    // ── Abbreviation expansion ────────────────────────────────────────────

    @Test
    fun `Mr dot Smith expands to mister smith`() {
        val result = TextNormalizer.normalize("Mr. Smith")
        assertEquals("mister smith", result)
    }

    @Test
    fun `Dr dot expands to doctor`() {
        val result = TextNormalizer.normalize("Dr. Watson")
        assertTrue(result.startsWith("doctor"))
    }

    @Test
    fun `Mrs dot expands to misses`() {
        val result = TextNormalizer.normalize("Mrs. Jones")
        assertTrue(result.startsWith("misses"))
    }

    @Test
    fun `St dot expands to saint`() {
        val result = TextNormalizer.normalize("St. Patrick")
        assertTrue(result.startsWith("saint"))
    }

    // ── Symbol expansion ──────────────────────────────────────────────────

    @Test
    fun `dollar five expands to five dollars`() {
        val result = TextNormalizer.normalize("$5")
        assertEquals("five dollars", result)
    }

    @Test
    fun `fifty percent expands correctly`() {
        val result = TextNormalizer.normalize("50%")
        assertEquals("fifty percent", result)
    }

    @Test
    fun `ampersand expands to and`() {
        val result = TextNormalizer.normalize("bread & butter")
        assertTrue(result.contains("and"))
    }

    // ── Basic normalization ───────────────────────────────────────────────

    @Test
    fun `uppercased input is lowercased`() {
        val result = TextNormalizer.normalize("HELLO WORLD")
        assertEquals("hello world", result)
    }

    @Test
    fun `multiple spaces collapsed to single space`() {
        val result = TextNormalizer.normalize("hello   world")
        assertEquals("hello world", result)
    }

    @Test
    fun `leading and trailing whitespace trimmed`() {
        val result = TextNormalizer.normalize("  hello  ")
        assertEquals("hello", result)
    }

    @Test
    fun `allowed punctuation is preserved`() {
        val result = TextNormalizer.normalize("Hello, world!")
        assertTrue(result.contains(","))
        assertTrue(result.contains("!"))
    }

    @Test
    fun `contractions preserved`() {
        // Apostrophe kept so CMU dict can handle "don't", "can't", etc.
        val result = TextNormalizer.normalize("don't stop")
        assertTrue(result.contains("don't"))
    }

    // ── Edge cases ────────────────────────────────────────────────────────

    @Test
    fun `empty string returns empty`() {
        assertEquals("", TextNormalizer.normalize(""))
    }

    @Test
    fun `blank string returns empty`() {
        assertEquals("", TextNormalizer.normalize("   "))
    }

    @Test
    fun `bare integer in sentence expanded`() {
        val result = TextNormalizer.normalize("I have 3 cats")
        assertTrue(result.contains("three"))
    }

    @Test
    fun `mixed abbreviation and number`() {
        val result = TextNormalizer.normalize("Dr. 2 cats")
        assertTrue(result.startsWith("doctor"))
        assertTrue(result.contains("two"))
    }
}
