package com.hereliesaz.liperty.voicebox.g2p

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Pure JUnit 4 tests for [CMUDictionary].
 * Uses inline compact-format data — no file I/O required.
 */
class CMUDictionaryTest {

    private lateinit var dict: CMUDictionary

    /** Minimal inline CMU compact corpus covering several test cases. */
    private val sampleText = """
        ;;; CMU Pronouncing Dictionary (compact test fixture)
        HELLO	HH AH0 L OW1
        WORLD	W ER1 L D
        HELLO(1)	HH EH0 L OW1
        CAT	K AE1 T
        DOG	D AO1 G
        THE	DH AH0

        QUICK	K W IH1 K
        ;;; another comment
        BROWN	B R AW1 N
    """.trimIndent()

    @Before
    fun setUp() {
        dict = CMUDictionary()
    }

    // ── isLoaded and size ─────────────────────────────────────────────────

    @Test
    fun `isLoaded false before loading`() {
        assertFalse(dict.isLoaded)
    }

    @Test
    fun `isLoaded true after valid load`() {
        dict.loadFromText(sampleText)
        assertTrue(dict.isLoaded)
    }

    @Test
    fun `size reflects number of primary entries`() {
        dict.loadFromText(sampleText)
        // Variants (HELLO(1)) and comment/blank lines must not be counted.
        assertEquals(7, dict.size)
    }

    // ── Lookup correctness ────────────────────────────────────────────────

    @Test
    fun `lookup known word returns correct phonemes`() {
        dict.loadFromText(sampleText)
        val result = dict.lookup("HELLO")
        assertNotNull(result)
        assertEquals(listOf("HH", "AH0", "L", "OW1"), result)
    }

    @Test
    fun `lookup another known word`() {
        dict.loadFromText(sampleText)
        val result = dict.lookup("CAT")
        assertNotNull(result)
        assertEquals(listOf("K", "AE1", "T"), result)
    }

    // ── Case insensitivity ────────────────────────────────────────────────

    @Test
    fun `lookup lowercase word matches uppercase entry`() {
        dict.loadFromText(sampleText)
        val lower = dict.lookup("hello")
        val upper = dict.lookup("HELLO")
        assertEquals(upper, lower)
    }

    @Test
    fun `lookup mixed-case word matches`() {
        dict.loadFromText(sampleText)
        val mixed = dict.lookup("Hello")
        assertNotNull(mixed)
        assertEquals(listOf("HH", "AH0", "L", "OW1"), mixed)
    }

    // ── OOV handling ──────────────────────────────────────────────────────

    @Test
    fun `lookup unknown word returns null`() {
        dict.loadFromText(sampleText)
        assertNull(dict.lookup("XYZNOTAWORD"))
    }

    @Test
    fun `lookup on empty dictionary returns null`() {
        assertNull(dict.lookup("HELLO"))
    }

    // ── Variant pronunciation skipping ────────────────────────────────────

    @Test
    fun `variant pronunciation HELLO(1) is not stored as separate entry`() {
        dict.loadFromText(sampleText)
        // Asking for the variant key directly should return null because we
        // only store the headword without the (N) suffix.
        assertNull(dict.lookup("HELLO(1)"))
    }

    @Test
    fun `first pronunciation wins for HELLO`() {
        dict.loadFromText(sampleText)
        // The canonical entry (no suffix) has "AH0" as the second phoneme.
        val phonemes = dict.lookup("HELLO")
        assertEquals("AH0", phonemes?.get(1))
    }

    // ── Comment and blank line skipping ───────────────────────────────────

    @Test
    fun `comment lines do not produce entries`() {
        dict.loadFromText(";;; this is a comment\nHELLO\tHH EH1 L OW1")
        // Size should be exactly 1 (just HELLO).
        assertEquals(1, dict.size)
    }

    @Test
    fun `blank lines are ignored`() {
        dict.loadFromText("\n\nHELLO\tHH EH1 L OW1\n\n")
        assertEquals(1, dict.size)
    }

    // ── Parsing edge cases ────────────────────────────────────────────────

    @Test
    fun `line without tab is skipped`() {
        dict.loadFromText("NOPHONE\nHELLO\tHH EH1 L OW1")
        assertNull(dict.lookup("NOPHONE"))
        assertNotNull(dict.lookup("HELLO"))
    }

    @Test
    fun `multiple loads merge entries`() {
        dict.loadFromText("CAT\tK AE1 T")
        dict.loadFromText("DOG\tD AO1 G")
        assertEquals(2, dict.size)
        assertNotNull(dict.lookup("CAT"))
        assertNotNull(dict.lookup("DOG"))
    }
}
