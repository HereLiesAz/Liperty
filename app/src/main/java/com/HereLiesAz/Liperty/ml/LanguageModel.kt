package com.HereLiesAz.Liperty.ml

class LanguageModel {

    // Simple Bi-gram map: "A" -> [("B", 0.1), ("C", 0.2)...]
    // Key: Previous Word/Char, Value: Map of Next Token to Probability
    private val bigramMap = mutableMapOf<String, Map<String, Double>>()

    init {
        // Initialize with dummy data
        // Ideally load from a file
        val commonPairs = mapOf(
            "H" to mapOf("E" to 0.5, "I" to 0.3),
            "E" to mapOf("L" to 0.4, "R" to 0.2),
            "L" to mapOf("L" to 0.4, "O" to 0.3),
            "_" to mapOf("H" to 0.1, "T" to 0.1) // Start of sentence
        )
        bigramMap.putAll(commonPairs)
    }

    fun getScore(prevToken: String, currentToken: String): Double {
        val nextMap = bigramMap[prevToken] ?: return 0.0
        return nextMap[currentToken] ?: 0.0
    }
}
