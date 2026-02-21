package com.HereLiesAz.Liperty.ml

class HomopheneCorrector {

    // Simplified Homophene Groups (Visually indistinguishable)
    // Based on research: p/b/m, f/v, t/d/n, etc.
    private val homopheneGroups = listOf(
        setOf("p", "b", "m"),
        setOf("f", "v"),
        setOf("t", "d", "n", "l"), // simplified
        setOf("k", "g", "ng"),
        setOf("s", "z"),
        setOf("sh", "ch", "j", "zh")
    )

    // Dictionary of words to their homophene alternatives
    // Key: Word, Value: List of words that look the same
    private val homopheneMap = mapOf(
        "mat" to listOf("bat", "pat"),
        "bat" to listOf("mat", "pat"),
        "pat" to listOf("mat", "bat"),
        "meet" to listOf("beat", "peat"),
        "beat" to listOf("meet", "peat"),
        "peat" to listOf("meet", "beat"),
        "time" to listOf("dime"),
        "dime" to listOf("time")
    )

    /**
     * Returns a list of alternative words that are homophenes of the input word.
     */
    fun getAlternatives(word: String): List<String> {
        val lowerCaseWord = word.lowercase()
        return homopheneMap[lowerCaseWord] ?: emptyList()
    }

    /**
     * More advanced: Check if two words are homophenes based on phoneme mapping.
     * (Placeholder for future implementation)
     */
    fun areHomophenes(word1: String, word2: String): Boolean {
        // Todo: Map words to visemes and compare
        val alts = getAlternatives(word1)
        return alts.contains(word2.lowercase())
    }
}
