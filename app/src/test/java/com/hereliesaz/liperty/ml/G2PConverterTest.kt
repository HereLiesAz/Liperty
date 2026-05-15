package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class G2PConverterTest {

    private val converter = G2PConverter()

    @Test
    fun testDictionaryWords() {
        // Test known dictionary words
        assertEquals(listOf("DH", "AH"), converter.wordToPhonemes("THE"))
        assertEquals(listOf("K", "W", "IH", "K"), converter.wordToPhonemes("QUICK"))
        assertEquals(listOf("D", "AO", "G"), converter.wordToPhonemes("DOG"))
    }

    @Test
    fun testFallbackRules() {
        // Test some basic digraphs and rule fallbacks not in the explicit dictionary list
        // "TEST" -> "T" "EH" "S" "T"
        assertEquals(listOf("T", "EH", "S", "T"), converter.wordToPhonemes("TEST"))
        
        // "CHIP" -> "CH" "IH" "P"
        assertEquals(listOf("CH", "IH", "P"), converter.wordToPhonemes("CHIP"))
        
        // "SHOP" -> "SH" "AA" "P"
        assertEquals(listOf("SH", "AA", "P"), converter.wordToPhonemes("SHOP"))
        
        // "THING" -> "TH" "IH" "NG"
        assertEquals(listOf("TH", "IH", "NG"), converter.wordToPhonemes("THING"))

        // Special case handling
        // X -> "K" "S" -> "MAX" -> "M" "AE" "K" "S"
        assertEquals(listOf("M", "AE", "K", "S"), converter.wordToPhonemes("MAX"))
    }

    @Test
    fun testSentenceProcessing() {
        // Test ignoring punctuation and case
        val sentence = "The quick, brown fox!"
        val expected = listOf(
            "DH", "AH",                   // THE
            "K", "W", "IH", "K",          // QUICK
            "B", "R", "AW", "N",          // BROWN
            "F", "AA", "K", "S"           // FOX
        )
        assertEquals(expected, converter.sentenceToPhonemes(sentence))
    }

    @Test
    fun testCalibrationPhrase() {
        val phrase = "Sphinx of black quartz judge my vow"
        val expected = listOf(
            "S", "F", "IH", "NG", "K", "S", // SPHINX
            "AH", "V",                      // OF
            "B", "L", "AE", "K",            // BLACK
            "K", "W", "AO", "R", "T", "S",  // QUARTZ
            "JH", "AH", "JH",               // JUDGE
            "M", "AY",                      // MY
            "V", "AW"                       // VOW
        )
        assertEquals(expected, converter.sentenceToPhonemes(phrase))
    }
}
