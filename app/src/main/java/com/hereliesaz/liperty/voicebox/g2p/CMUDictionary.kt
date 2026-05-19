package com.hereliesaz.liperty.voicebox.g2p

/**
 * Compact CMU Pronouncing Dictionary loader.
 *
 * Accepts text in the tab-separated format produced by the project's
 * `tools/build_cmu_compact.py` script:
 *
 *     WORD<TAB>PH1 PH2 PH3
 *     ...
 *
 * Lines beginning with `;;;` are comments and are skipped, as are blank
 * lines and variant-pronunciation entries of the form `WORD(N)`.
 *
 * Pure Kotlin — no Android dependencies.
 */
class CMUDictionary {

    private val dict = HashMap<String, List<String>>(120_000)

    /** True once [loadFromText] has completed successfully with at least one entry. */
    val isLoaded: Boolean get() = dict.isNotEmpty()

    /** Number of primary entries currently loaded. */
    val size: Int get() = dict.size

    /**
     * Parses [text] in compact CMU format and populates the internal map.
     * May be called multiple times to merge additional entries; later entries
     * for the same word override earlier ones.
     *
     * Variant-pronunciation entries — those whose headword ends with `(N)` —
     * are silently skipped, keeping only the first (canonical) pronunciation.
     */
    fun loadFromText(text: String) {
        for (rawLine in text.lineSequence()) {
            val line = rawLine.trim()
            if (line.isEmpty() || line.startsWith(";;;")) continue

            val tabIndex = line.indexOf('\t')
            if (tabIndex < 1) continue          // no tab — skip malformed line

            val headword = line.substring(0, tabIndex)

            // Skip variant pronunciations: WORD(1), WORD(2), …
            if (headword.endsWith(')') && headword.contains('(')) continue

            val phonemeStr = line.substring(tabIndex + 1).trim()
            if (phonemeStr.isEmpty()) continue

            val phonemes = phonemeStr.split(' ').filter { it.isNotEmpty() }
            if (phonemes.isEmpty()) continue

            dict[headword.uppercase()] = phonemes
        }
    }

    /**
     * Returns the ARPABET phoneme list (with stress markers, e.g. "AE1") for [word],
     * or `null` if the word is not in the dictionary.
     *
     * Lookup is case-insensitive.
     */
    fun lookup(word: String): List<String>? = dict[word.uppercase()]
}
