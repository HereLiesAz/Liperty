package com.hereliesaz.liperty.voicebox

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * Tests for [VitsTokenizer] — parses `pocket_tts_phoneme_map.json` and
 * converts English text to VITS input_ids (with blank interleaving).
 *
 * Pure JUnit 4 — no Robolectric or Android Context needed.
 */
class VitsTokenizerTest {

    private lateinit var tokenizer: VitsTokenizer

    /** Minimal phoneme-model JSON with IPA symbols and char_to_id. */
    private val phonemeModelJson = """
    {
      "is_phoneme": true,
      "add_blank": true,
      "blank": "<blnk>",
      "pad": "<pad>",
      "char_to_id": {
        "<blnk>": 0,
        "<pad>": 1,
        " ": 2,
        "h": 3,
        "ɛ": 4,
        "l": 5,
        "oʊ": 6,
        "o": 7,
        "ʊ": 8,
        "w": 9,
        "ɝ": 10,
        "d": 11,
        "ð": 12,
        "ʌ": 13,
        "k": 14,
        "ɪ": 15
      },
      "characters": [],
      "phonemes": ["<blnk>", "<pad>", " ", "h", "ɛ", "l", "o", "ʊ", "w", "ɝ", "d", "ð", "ʌ", "k", "ɪ"],
      "sample_rate": 22050
    }
    """.trimIndent()

    /** Character-model JSON (is_phoneme=false). */
    private val charModelJson = """
    {
      "is_phoneme": false,
      "add_blank": true,
      "blank": "<blnk>",
      "char_to_id": {
        "<blnk>": 0,
        " ": 1,
        "a": 2,
        "b": 3,
        "c": 4,
        "d": 5,
        "e": 6,
        "h": 7,
        "l": 8,
        "o": 9,
        "r": 10,
        "w": 11
      },
      "characters": ["<blnk>", " ", "a", "b", "c", "d", "e", "h", "l", "o", "r", "w"],
      "phonemes": [],
      "sample_rate": 22050
    }
    """.trimIndent()

    @Before
    fun setUp() {
        tokenizer = VitsTokenizer()
    }

    // ── Loading ──────────────────────────────────────────────────────────

    @Test
    fun `isLoaded is false before loading`() {
        assertFalse(tokenizer.isLoaded)
    }

    @Test
    fun `loads phoneme model JSON successfully`() {
        assertTrue(tokenizer.loadFromJson(phonemeModelJson))
        assertTrue(tokenizer.isLoaded)
    }

    @Test
    fun `loads character model JSON successfully`() {
        assertTrue(tokenizer.loadFromJson(charModelJson))
        assertTrue(tokenizer.isLoaded)
    }

    @Test
    fun `returns false for invalid JSON`() {
        assertFalse(tokenizer.loadFromJson("not json"))
        assertFalse(tokenizer.isLoaded)
    }

    @Test
    fun `returns false for JSON without char_to_id`() {
        assertFalse(tokenizer.loadFromJson("""{"is_phoneme": true}"""))
    }

    @Test
    fun `returns false for empty char_to_id`() {
        assertFalse(tokenizer.loadFromJson("""{"char_to_id": {}}"""))
    }

    // ── Character-level tokenization ─────────────────────────────────────

    @Test
    fun `character model tokenizes lowercase with blanks`() {
        tokenizer.loadFromJson(charModelJson)
        // "hello" → h(7) e(6) l(8) l(8) o(9) with blanks (0) between
        val tokens = tokenizer.tokenize("hello")
        // Expected: [0, 7, 0, 6, 0, 8, 0, 8, 0, 9, 0]
        assertArrayEquals(longArrayOf(0, 7, 0, 6, 0, 8, 0, 8, 0, 9, 0), tokens)
    }

    @Test
    fun `character model skips unknown characters`() {
        tokenizer.loadFromJson(charModelJson)
        // "abc" → a(2) b(3) c(4); no mapping for z
        val tokens = tokenizer.tokenize("abcz")
        assertArrayEquals(longArrayOf(0, 2, 0, 3, 0, 4, 0), tokens)
    }

    @Test
    fun `character model converts uppercase to lowercase`() {
        tokenizer.loadFromJson(charModelJson)
        val upper = tokenizer.tokenize("HELLO")
        val lower = tokenizer.tokenize("hello")
        assertArrayEquals(lower, upper)
    }

    @Test
    fun `empty text produces empty tokens`() {
        tokenizer.loadFromJson(charModelJson)
        val tokens = tokenizer.tokenize("")
        // Only the initial blank remains
        assertArrayEquals(longArrayOf(0), tokens)
    }

    @Test
    fun `tokenize returns empty array when not loaded`() {
        val tokens = tokenizer.tokenize("hello")
        assertEquals(0, tokens.size)
    }

    // ── No-blank mode ────────────────────────────────────────────────────

    @Test
    fun `no blanks when add_blank is false`() {
        val json = """
        {
          "is_phoneme": false,
          "add_blank": false,
          "blank": "<blnk>",
          "char_to_id": { "<blnk>": 0, "a": 1, "b": 2 }
        }
        """.trimIndent()
        tokenizer.loadFromJson(json)
        val tokens = tokenizer.tokenize("ab")
        assertArrayEquals(longArrayOf(1, 2), tokens)
    }

    // ── IPA conversion (phoneme model) ───────────────────────────────────

    @Test
    fun `textToIpa converts English to IPA`() {
        tokenizer.loadFromJson(phonemeModelJson)
        // "the" → G2PConverter: THE → [DH, AH] → IPA: [ð, ʌ] → "ðʌ"
        val ipa = tokenizer.textToIpa("the")
        assertEquals("ðʌ", ipa)
    }

    @Test
    fun `textToIpa separates words with space`() {
        tokenizer.loadFromJson(phonemeModelJson)
        val ipa = tokenizer.textToIpa("the quick")
        // THE → "ðʌ", QUICK → "kwɪk"
        assertTrue(ipa.contains(" "))
        assertTrue(ipa.startsWith("ð"))
    }

    @Test
    fun `textToIpa strips non-alphabetic characters`() {
        tokenizer.loadFromJson(phonemeModelJson)
        val withPunctuation = tokenizer.textToIpa("the!")
        val withoutPunctuation = tokenizer.textToIpa("the")
        assertEquals(withoutPunctuation, withPunctuation)
    }
}
