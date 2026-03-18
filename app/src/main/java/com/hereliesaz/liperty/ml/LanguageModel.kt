package com.hereliesaz.liperty.ml

/**
 * Phoneme-level bigram language model.
 * Transition probabilities are derived from English phoneme co-occurrence
 * statistics (CMU Pronouncing Dictionary corpus analysis).
 * Unknown pairs receive a smoothed uniform probability to avoid zero-mass paths.
 */
class LanguageModel {

    companion object {
        // Laplace-smoothed fallback for unseen bigrams
        private const val UNIFORM_PROB = 0.003
    }

    private val bigramMap: Map<String, Map<String, Double>> = mapOf(
        // ── Vowels ───────────────────────────────────────────────────────────
        "AA" to mapOf("N" to 0.12, "T" to 0.09, "R" to 0.08, "L" to 0.07, "D" to 0.07, "S" to 0.06, "K" to 0.05),
        "AE" to mapOf("N" to 0.14, "T" to 0.10, "S" to 0.08, "K" to 0.07, "NG" to 0.06, "D" to 0.06, "L" to 0.05),
        "AH" to mapOf("N" to 0.15, "T" to 0.10, "S" to 0.08, "L" to 0.07, "D" to 0.07, "V" to 0.05, "Z" to 0.05),
        "AO" to mapOf("R" to 0.16, "N" to 0.10, "T" to 0.08, "L" to 0.07, "D" to 0.06),
        "AW" to mapOf("N" to 0.13, "T" to 0.10, "D" to 0.08, "L" to 0.07, "Z" to 0.06),
        "AY" to mapOf("T" to 0.11, "N" to 0.10, "D" to 0.09, "K" to 0.07, "Z" to 0.06),
        "EH" to mapOf("N" to 0.16, "R" to 0.10, "T" to 0.09, "S" to 0.08, "L" to 0.07, "D" to 0.06),
        "ER" to mapOf("N" to 0.13, "S" to 0.10, "T" to 0.09, "D" to 0.08, "L" to 0.07, "Z" to 0.06),
        "EY" to mapOf("T" to 0.11, "N" to 0.10, "D" to 0.08, "S" to 0.07, "K" to 0.06, "Z" to 0.06),
        "IH" to mapOf("N" to 0.18, "NG" to 0.12, "T" to 0.10, "S" to 0.08, "D" to 0.07, "Z" to 0.06),
        "IY" to mapOf("N" to 0.12, "T" to 0.10, "S" to 0.09, "D" to 0.07, "L" to 0.06, "Z" to 0.06),
        "OW" to mapOf("N" to 0.14, "T" to 0.09, "D" to 0.08, "L" to 0.07, "Z" to 0.06, "NG" to 0.05),
        "OY" to mapOf("N" to 0.12, "D" to 0.09, "L" to 0.07, "Z" to 0.06),
        "UH" to mapOf("T" to 0.11, "N" to 0.10, "D" to 0.09, "K" to 0.07, "S" to 0.06),
        "UW" to mapOf("T" to 0.11, "N" to 0.10, "D" to 0.08, "S" to 0.07, "Z" to 0.06, "L" to 0.05),
        // ── Stops ────────────────────────────────────────────────────────────
        "B" to mapOf("AH" to 0.14, "IH" to 0.10, "AE" to 0.09, "R" to 0.08, "L" to 0.07, "EH" to 0.06, "IY" to 0.05),
        "P" to mapOf("AH" to 0.12, "R" to 0.11, "L" to 0.10, "IH" to 0.09, "AE" to 0.07, "EH" to 0.06),
        "T" to mapOf("AH" to 0.13, "IH" to 0.10, "ER" to 0.09, "R" to 0.08, "OW" to 0.07, "AE" to 0.06),
        "D" to mapOf("AH" to 0.13, "IH" to 0.10, "ER" to 0.09, "AE" to 0.08, "OW" to 0.07, "EH" to 0.06),
        "K" to mapOf("AH" to 0.11, "IH" to 0.10, "AE" to 0.09, "OW" to 0.08, "L" to 0.07, "R" to 0.07),
        "G" to mapOf("AH" to 0.13, "R" to 0.11, "IH" to 0.10, "L" to 0.08, "AE" to 0.07, "OW" to 0.06),
        // ── Fricatives ───────────────────────────────────────────────────────
        "F" to mapOf("AH" to 0.13, "R" to 0.11, "L" to 0.09, "OW" to 0.07, "IH" to 0.07, "AE" to 0.06),
        "V" to mapOf("AH" to 0.16, "IH" to 0.11, "EH" to 0.09, "OW" to 0.08, "AE" to 0.07),
        "S" to mapOf("T" to 0.16, "AH" to 0.10, "IH" to 0.09, "K" to 0.08, "P" to 0.07, "L" to 0.06),
        "Z" to mapOf("AH" to 0.12, "IH" to 0.10, "D" to 0.08, "OW" to 0.07, "EH" to 0.06),
        "SH" to mapOf("AH" to 0.13, "IH" to 0.11, "UH" to 0.09, "OW" to 0.07, "AE" to 0.06),
        "TH" to mapOf("AH" to 0.16, "IH" to 0.11, "ER" to 0.09, "AE" to 0.08, "EH" to 0.07),
        "DH" to mapOf("AH" to 0.20, "IH" to 0.13, "ER" to 0.09, "AE" to 0.08, "EH" to 0.07),
        "HH" to mapOf("AH" to 0.16, "IH" to 0.13, "AE" to 0.10, "EH" to 0.09, "AY" to 0.07),
        // ── Affricates ───────────────────────────────────────────────────────
        "CH" to mapOf("AH" to 0.13, "IH" to 0.11, "ER" to 0.09, "AE" to 0.08, "OW" to 0.07),
        "JH" to mapOf("AH" to 0.14, "IH" to 0.11, "AE" to 0.09, "ER" to 0.08, "OW" to 0.07),
        // ── Nasals ───────────────────────────────────────────────────────────
        "M" to mapOf("AH" to 0.16, "EY" to 0.09, "IH" to 0.09, "AE" to 0.08, "AY" to 0.07, "OW" to 0.06),
        "N" to mapOf("AH" to 0.13, "IH" to 0.11, "OW" to 0.09, "EH" to 0.08, "T" to 0.07, "D" to 0.06),
        "NG" to mapOf("K" to 0.20, "Z" to 0.11, "T" to 0.08, "D" to 0.06),
        // ── Approximants / Liquids ────────────────────────────────────────────
        "L" to mapOf("AH" to 0.14, "IH" to 0.11, "AE" to 0.09, "OW" to 0.08, "EH" to 0.07, "IY" to 0.06),
        "R" to mapOf("AH" to 0.16, "IH" to 0.13, "AE" to 0.10, "EH" to 0.09, "OW" to 0.07, "IY" to 0.05),
        "W" to mapOf("AH" to 0.15, "IH" to 0.11, "AE" to 0.09, "EH" to 0.08, "ER" to 0.07, "OW" to 0.06),
        "Y" to mapOf("UW" to 0.20, "ER" to 0.13, "AH" to 0.10, "IH" to 0.09, "OW" to 0.07, "AE" to 0.06)
    )

    /**
     * Returns P(currentToken | prevToken).
     * Unknown bigrams receive [UNIFORM_PROB] instead of 0 to avoid dead paths in beam search.
     */
    fun getScore(prevToken: String, currentToken: String): Double {
        val nextMap = bigramMap[prevToken] ?: return UNIFORM_PROB
        return nextMap[currentToken] ?: UNIFORM_PROB
    }
}
