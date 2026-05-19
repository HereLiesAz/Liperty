package com.hereliesaz.liperty.voicebox.g2p

import android.app.Application
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/**
 * Tests for [MeloTTSTokenizer].
 *
 * The G2POnnxPredictor (which requires a live ONNX session) is NOT loaded here —
 * tests exercise the CMU dict path and the rule-based fallback path only.
 *
 * Runs under Robolectric because [MeloTTSTokenizer.loadVocab] uses
 * `org.json.JSONObject`, which is a stub in pure-JVM unit tests.
 *
 * Robolectric quirks:
 * - `@Config(sdk=[34])` works around Robolectric not yet supporting
 *   `targetSdk=37`; see CLAUDE.md "Robolectric SDK 37 trap".
 * - `application = android.app.Application::class` substitutes a stub
 *   Application for [com.hereliesaz.liperty.LipertyApplication], which
 *   would otherwise crash on `BluetoothAdapter.getProfileProxy()` during
 *   Robolectric setup. The tokenizer doesn't need any Application state,
 *   so an empty stub is sufficient.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34], application = Application::class)
class MeloTTSTokenizerTest {

    private lateinit var tokenizer: MeloTTSTokenizer

    /**
     * Minimal `pocket_tts_vocab.json` fixture.
     *
     * Symbols chosen to cover the rule-based fallback output for "hello" and
     * "world" (H→HH, E→EH, L→L, O→AA, W→W, R→R, D→D) plus punctuation.
     *
     * arpabet_to_symbol keys use bare (no stress digit) forms for simplicity;
     * the tokenizer tries the full form first, then the bare form.
     */
    private val vocabJson = """
    {
      "vocab": {
        "HH": 1,
        "EH": 2,
        "L":  3,
        "AA": 4,
        "W":  5,
        "ER": 6,
        "D":  7,
        "K":  8,
        "IH": 9,
        "AH": 10,
        "OW": 11,
        "AE": 12,
        "T":  13,
        ",":  20,
        "!":  21,
        ".":  22
      },
      "arpabet_to_symbol": {
        "HH":  "HH",
        "EH":  "EH",
        "EH0": "EH",
        "EH1": "EH",
        "L":   "L",
        "AA":  "AA",
        "AA0": "AA",
        "AA1": "AA",
        "W":   "W",
        "ER":  "ER",
        "ER0": "ER",
        "ER1": "ER",
        "D":   "D",
        "K":   "K",
        "IH":  "IH",
        "IH0": "IH",
        "IH1": "IH",
        "AH":  "AH",
        "AH0": "AH",
        "AH1": "AH",
        "OW":  "OW",
        "OW0": "OW",
        "OW1": "OW",
        "AE":  "AE",
        "AE1": "AE",
        "T":   "T"
      },
      "g2p_grapheme_vocab": ["<pad>", "a", "b", "c", "d", "e", "f", "g", "h", "i",
                              "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                              "t", "u", "v", "w", "x", "y", "z"],
      "g2p_phoneme_vocab":  ["<blank>", "AH0", "AE1", "L", "D", "T", "K", "IH1", "ER1", "HH", "EH1"]
    }
    """.trimIndent()

    /**
     * Minimal CMU dict text containing HELLO and WORLD so the dict path
     * is exercised for those words.
     */
    private val cmuText = """
        HELLO	HH AH0 L OW1
        WORLD	W ER1 L D
        QUICK	K W IH1 K
    """.trimIndent()

    @Before
    fun setUp() {
        tokenizer = MeloTTSTokenizer()
    }

    // ── Loading ──────────────────────────────────────────────────────────

    @Test
    fun `isLoaded is false before loading`() {
        assertFalse(tokenizer.isLoaded)
    }

    @Test
    fun `loadVocab returns true for valid JSON`() {
        assertTrue(tokenizer.loadVocab(vocabJson))
        assertTrue(tokenizer.isLoaded)
    }

    @Test
    fun `loadVocab returns false for invalid JSON`() {
        assertFalse(tokenizer.loadVocab("not json"))
    }

    @Test
    fun `loadVocab returns false for JSON without vocab object`() {
        assertFalse(tokenizer.loadVocab("""{"arpabet_to_symbol": {}}"""))
    }

    @Test
    fun `graphemeVocab exposed after load`() {
        tokenizer.loadVocab(vocabJson)
        assertTrue(tokenizer.graphemeVocab.isNotEmpty())
    }

    @Test
    fun `phonemeVocab exposed after load`() {
        tokenizer.loadVocab(vocabJson)
        assertTrue(tokenizer.phonemeVocab.isNotEmpty())
    }

    // ── tokenize: null / empty guards ────────────────────────────────────

    @Test
    fun `tokenize returns null when not loaded`() {
        assertNull(tokenizer.tokenize("hello"))
    }

    @Test
    fun `tokenize returns null for empty input`() {
        tokenizer.loadVocab(vocabJson)
        assertNull(tokenizer.tokenize(""))
    }

    @Test
    fun `tokenize returns null for blank input`() {
        tokenizer.loadVocab(vocabJson)
        assertNull(tokenizer.tokenize("   "))
    }

    // ── tokenize: dictionary path ─────────────────────────────────────────

    @Test
    fun `known word via CMU dict produces non-empty token array`() {
        tokenizer.loadVocab(vocabJson)
        val cmu = CMUDictionary().also { it.loadFromText(cmuText) }
        tokenizer.setCmuDictionary(cmu)

        val result = tokenizer.tokenize("hello")
        assertNotNull(result)
        assertTrue(result!!.isNotEmpty())
    }

    @Test
    fun `hello world via CMU dict produces tokens for both words`() {
        tokenizer.loadVocab(vocabJson)
        val cmu = CMUDictionary().also { it.loadFromText(cmuText) }
        tokenizer.setCmuDictionary(cmu)

        val result = tokenizer.tokenize("hello world")
        assertNotNull(result)
        // HELLO: HH(1) AH(10) L(3) OW(11) = 4 tokens
        // WORLD: W(5) ER(6) L(3) D(7)     = 4 tokens
        // Total = 8
        assertEquals(8, result!!.size)
    }

    // ── tokenize: punctuation ─────────────────────────────────────────────

    @Test
    fun `comma produces token when in vocab`() {
        tokenizer.loadVocab(vocabJson)
        val result = tokenizer.tokenize(",")
        assertNotNull(result)
        assertEquals(1, result!!.size)
        assertEquals(20L, result[0])
    }

    @Test
    fun `punctuation in sentence produces additional tokens`() {
        tokenizer.loadVocab(vocabJson)
        val cmu = CMUDictionary().also { it.loadFromText(cmuText) }
        tokenizer.setCmuDictionary(cmu)

        // "hello," should yield hello tokens + comma token.
        val withPunct = tokenizer.tokenize("hello,")
        val withoutPunct = tokenizer.tokenize("hello")

        assertNotNull(withPunct)
        assertNotNull(withoutPunct)
        assertTrue(withPunct!!.size > withoutPunct!!.size)
    }

    // ── tokenize: rule-based fallback (OOV, no neural G2P) ───────────────

    @Test
    fun `OOV word with no CMU dict falls back to rule-based`() {
        // Load vocab but do NOT attach a CMU dictionary.
        tokenizer.loadVocab(vocabJson)

        // "cat" → C→K(8), A→AE(12), T→T(13) via rule-based.
        val result = tokenizer.tokenize("cat")
        assertNotNull(result)
        assertTrue(result!!.isNotEmpty())
    }

    @Test
    fun `rule-based fallback for single letter word`() {
        tokenizer.loadVocab(vocabJson)
        // "d" → D(7) via rule-based single-char mapping.
        val result = tokenizer.tokenize("d")
        assertNotNull(result)
        assertArrayEquals(longArrayOf(7L), result)
    }

    // ── wordToArpabet: internal behavior ─────────────────────────────────

    @Test
    fun `wordToArpabet returns CMU dict entry when dict attached`() {
        tokenizer.loadVocab(vocabJson)
        val cmu = CMUDictionary().also { it.loadFromText(cmuText) }
        tokenizer.setCmuDictionary(cmu)

        val phonemes = tokenizer.wordToArpabet("hello")
        assertEquals(listOf("HH", "AH0", "L", "OW1"), phonemes)
    }

    @Test
    fun `wordToArpabet falls back to rule-based when dict has no entry`() {
        tokenizer.loadVocab(vocabJson)
        // No dict attached; rule-based for "cat" → K AE T.
        val phonemes = tokenizer.wordToArpabet("cat")
        assertEquals(listOf("K", "AE", "T"), phonemes)
    }

    @Test
    fun `wordToArpabet handles digraph CH`() {
        tokenizer.loadVocab(vocabJson)
        val phonemes = tokenizer.wordToArpabet("chip")
        assertTrue(phonemes.first() == "CH")
    }

    @Test
    fun `wordToArpabet handles X as K S`() {
        tokenizer.loadVocab(vocabJson)
        val phonemes = tokenizer.wordToArpabet("max")
        // M AE K S
        assertTrue(phonemes.contains("K"))
        assertTrue(phonemes.contains("S"))
        val kIdx = phonemes.indexOf("K")
        val sIdx = phonemes.indexOf("S")
        assertTrue(kIdx < sIdx)
    }

    // ── ruleBasedG2P: direct tests ────────────────────────────────────────

    @Test
    fun `ruleBasedG2P digraphs`() {
        tokenizer.loadVocab(vocabJson)
        assertEquals(listOf("CH", "IH", "P"), tokenizer.ruleBasedG2P("CHIP"))
        assertEquals(listOf("SH", "AA", "P"), tokenizer.ruleBasedG2P("SHOP"))
        assertEquals(listOf("TH", "IH", "NG"), tokenizer.ruleBasedG2P("THING"))
        assertEquals(listOf("F", "AA", "N"), tokenizer.ruleBasedG2P("PHON"))
        assertEquals(listOf("K", "W", "IH", "K"), tokenizer.ruleBasedG2P("QUIK"))
        assertEquals(listOf("NG"), tokenizer.ruleBasedG2P("NG"))
        assertEquals(listOf("IY", "T"), tokenizer.ruleBasedG2P("EET"))
        assertEquals(listOf("UW", "T"), tokenizer.ruleBasedG2P("OOT"))
    }

    @Test
    fun `ruleBasedG2P single chars`() {
        tokenizer.loadVocab(vocabJson)
        assertEquals(listOf("AE"), tokenizer.ruleBasedG2P("A"))
        assertEquals(listOf("EH"), tokenizer.ruleBasedG2P("E"))
        assertEquals(listOf("IH"), tokenizer.ruleBasedG2P("I"))
        assertEquals(listOf("HH"), tokenizer.ruleBasedG2P("H"))
        assertEquals(listOf("K", "S"), tokenizer.ruleBasedG2P("X"))
    }
}
