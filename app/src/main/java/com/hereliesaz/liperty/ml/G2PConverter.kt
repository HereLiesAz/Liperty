package com.hereliesaz.liperty.ml

import java.util.Locale

/**
 * Grapheme-to-Phoneme converter for mapping English text to CMUdict-style phonemes.
 * Uses a built-in dictionary for common words and calibration phrases, with a rule-based fallback
 * for out-of-vocabulary words.
 */
class G2PConverter {

    // Small dictionary covering common words and specific calibration phrases
    private val dictionary = mapOf(
        "THE" to listOf("DH", "AH"),
        "QUICK" to listOf("K", "W", "IH", "K"),
        "BROWN" to listOf("B", "R", "AW", "N"),
        "FOX" to listOf("F", "AA", "K", "S"),
        "JUMPS" to listOf("JH", "AH", "M", "P", "S"),
        "OVER" to listOf("OW", "V", "ER"),
        "LAZY" to listOf("L", "EY", "Z", "IY"),
        "DOG" to listOf("D", "AO", "G"),
        "PACK" to listOf("P", "AE", "K"),
        "MY" to listOf("M", "AY"),
        "BOX" to listOf("B", "AA", "K", "S"),
        "WITH" to listOf("W", "IH", "DH"),
        "FIVE" to listOf("F", "AY", "V"),
        "DOZEN" to listOf("D", "AH", "Z", "AH", "N"),
        "LIQUOR" to listOf("L", "IH", "K", "ER"),
        "JUGS" to listOf("JH", "AH", "G", "Z"),
        "HOW" to listOf("HH", "AW"),
        "VEXINGLY" to listOf("V", "EH", "K", "S", "IH", "NG", "L", "IY"),
        "DAFT" to listOf("D", "AE", "F", "T"),
        "ZEBRAS" to listOf("Z", "IY", "B", "R", "AH", "Z"),
        "JUMP" to listOf("JH", "AH", "M", "P"),
        "SPHINX" to listOf("S", "F", "IH", "NG", "K", "S"),
        "OF" to listOf("AH", "V"),
        "BLACK" to listOf("B", "L", "AE", "K"),
        "QUARTZ" to listOf("K", "W", "AO", "R", "T", "S"),
        "JUDGE" to listOf("JH", "AH", "JH"),
        "VOW" to listOf("V", "AW"),
        "A" to listOf("EY"),
        "I" to listOf("AY"),
        "TO" to listOf("T", "UW"),
        "IS" to listOf("IH", "Z"),
        "IN" to listOf("IH", "N"),
        "IT" to listOf("IH", "T"),
        "YOU" to listOf("Y", "UW"),
        "THAT" to listOf("DH", "AE", "T"),
        "AND" to listOf("AE", "N", "D"),
        "ARE" to listOf("AA", "R")
    )

    /**
     * Converts a full sentence into a flat list of phonemes.
     * Non-alphabetic characters are ignored (they do not produce blanks here, handle punctuation externally if needed).
     */
    fun sentenceToPhonemes(sentence: String): List<String> {
        val words = sentence.uppercase(Locale.US).split(Regex("\\s+"))
        val phonemes = mutableListOf<String>()
        
        for (word in words) {
            val cleanWord = word.replace(Regex("[^A-Z]"), "")
            if (cleanWord.isNotEmpty()) {
                phonemes.addAll(wordToPhonemes(cleanWord))
            }
        }
        return phonemes
    }

    /**
     * Converts a single word to a list of phonemes using the dictionary.
     * Falls back to a simplistic rule-based approach if not in the dictionary.
     */
    fun wordToPhonemes(word: String): List<String> {
        val upperWord = word.uppercase(Locale.US)
        
        // Lookup in predefined dictionary
        dictionary[upperWord]?.let { return it }

        // Fallback rule-based G2P (very simplistic approach for demo/fallback)
        return ruleBasedG2P(upperWord)
    }

    private fun ruleBasedG2P(word: String): List<String> {
        val phonemes = mutableListOf<String>()
        var i = 0
        
        while (i < word.length) {
            val char = word[i]
            
            // Basic two-character rules
            if (i < word.length - 1) {
                val digraph = word.substring(i, i + 2)
                when (digraph) {
                    "CH" -> { phonemes.add("CH"); i += 2; continue }
                    "SH" -> { phonemes.add("SH"); i += 2; continue }
                    "TH" -> { phonemes.add("TH"); i += 2; continue }
                    "PH" -> { phonemes.add("F"); i += 2; continue }
                    "QU" -> { phonemes.add("K"); phonemes.add("W"); i += 2; continue }
                    "NG" -> { phonemes.add("NG"); i += 2; continue }
                    "EE" -> { phonemes.add("IY"); i += 2; continue }
                    "OO" -> { phonemes.add("UW"); i += 2; continue }
                }
            }

            // Fallback single character rules mapped to MLConstants.PHONEME_VOCAB
            val p = when (char) {
                'A' -> "AE"
                'B' -> "B"
                'C' -> "K"
                'D' -> "D"
                'E' -> "EH"
                'F' -> "F"
                'G' -> "G"
                'H' -> "HH"
                'I' -> "IH"
                'J' -> "JH"
                'K' -> "K"
                'L' -> "L"
                'M' -> "M"
                'N' -> "N"
                'O' -> "AA"
                'P' -> "P"
                'Q' -> "K"
                'R' -> "R"
                'S' -> "S"
                'T' -> "T"
                'U' -> "AH"
                'V' -> "V"
                'W' -> "W"
                'X' -> { phonemes.add("K"); "S" }
                'Y' -> "Y"
                'Z' -> "Z"
                else -> ""
            }
            
            if (p.isNotEmpty()) phonemes.add(p)
            i++
        }
        
        return phonemes
    }
}
