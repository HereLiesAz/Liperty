package com.hereliesaz.liperty.voicebox.g2p

import org.json.JSONObject

/**
 * Orchestrates text-to-token-id conversion for the MeloTTS / OpenVoice v2 base TTS model.
 *
 * Pipeline:
 * 1. [TextNormalizer.normalize] — expand numbers, abbreviations, symbols; lowercase.
 * 2. Split into word and punctuation tokens.
 * 3. For each word, obtain ARPABET phonemes via (in priority order):
 *    a. [CMUDictionary] — dictionary lookup.
 *    b. [G2POnnxPredictor] — neural G2P (if loaded).
 *    c. Letter-by-letter rule fallback (replicates [com.hereliesaz.liperty.ml.G2PConverter.ruleBasedG2P]).
 * 4. Map ARPABET phonemes → MeloTTS symbol strings via [arpabetToSymbol], then to token IDs
 *    via [symbolToId].
 * 5. Punctuation tokens are looked up directly in [symbolToId].
 * 6. Return a [LongArray] of token IDs, or `null` if the result is empty.
 *
 * Dependencies are injected for testability — no Android Context is required.
 *
 * Expected JSON structure of `pocket_tts_vocab.json`:
 * ```json
 * {
 *   "vocab": { "symbol": id, ... },
 *   "arpabet_to_symbol": { "AH0": "ə", ... },
 *   "g2p_grapheme_vocab": ["<pad>", "a", "b", ...],
 *   "g2p_phoneme_vocab":  ["<blank>", "AH0", "AE1", ...]
 * }
 * ```
 *
 * The `vocab` object must contain every MeloTTS symbol including punctuation
 * characters the model was trained with. Unknown symbols are silently skipped.
 */
class MeloTTSTokenizer {

    private var symbolToId: Map<String, Int> = emptyMap()
    private var arpabetToSymbol: Map<String, String> = emptyMap()

    private var cmuDict: CMUDictionary? = null
    private var neuralG2P: G2POnnxPredictor? = null

    /** Grapheme vocab exposed so [com.hereliesaz.liperty.voicebox.PocketTTSEngine] can pass it to [G2POnnxPredictor]. */
    var graphemeVocab: List<String> = emptyList()
        private set

    /** Phoneme vocab exposed for the same reason. */
    var phonemeVocab: List<String> = emptyList()
        private set

    /** True once [loadVocab] has succeeded. */
    val isLoaded: Boolean get() = symbolToId.isNotEmpty()

    // ── Initialisation ────────────────────────────────────────────────────

    /**
     * Parses `pocket_tts_vocab.json` and populates internal lookup tables.
     *
     * @return `true` on success, `false` if the JSON is malformed or the
     *         `vocab` object is missing / empty.
     */
    fun loadVocab(jsonString: String): Boolean {
        return try {
            val root = JSONObject(jsonString)

            val vocabObj = root.optJSONObject("vocab") ?: return false
            val built = buildMap<String, Int> {
                val keys = vocabObj.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    put(key, vocabObj.getInt(key))
                }
            }
            if (built.isEmpty()) return false
            symbolToId = built

            val arpObj = root.optJSONObject("arpabet_to_symbol")
            arpabetToSymbol = if (arpObj != null) {
                buildMap {
                    val keys = arpObj.keys()
                    while (keys.hasNext()) {
                        val key = keys.next()
                        put(key, arpObj.getString(key))
                    }
                }
            } else emptyMap()

            val graphArr = root.optJSONArray("g2p_grapheme_vocab")
            graphemeVocab = if (graphArr != null) {
                (0 until graphArr.length()).map { graphArr.getString(it) }
            } else emptyList()

            val phonArr = root.optJSONArray("g2p_phoneme_vocab")
            phonemeVocab = if (phonArr != null) {
                (0 until phonArr.length()).map { phonArr.getString(it) }
            } else emptyList()

            true
        } catch (_: Exception) {
            false
        }
    }

    /** Attaches a loaded [CMUDictionary]. May be called after [loadVocab]. */
    fun setCmuDictionary(dict: CMUDictionary) {
        cmuDict = dict
    }

    /** Attaches a loaded [G2POnnxPredictor]. May be called after [loadVocab]. */
    fun setNeuralG2P(predictor: G2POnnxPredictor) {
        neuralG2P = predictor
    }

    // ── Tokenization pipeline ─────────────────────────────────────────────

    /**
     * Converts [text] to a [LongArray] of MeloTTS token IDs.
     *
     * Returns `null` if [isLoaded] is false or if no tokens can be produced
     * (e.g. the text contains only symbols not in the vocab).
     */
    fun tokenize(text: String): LongArray? {
        if (!isLoaded) return null

        val normalized = TextNormalizer.normalize(text)
        if (normalized.isBlank()) return null

        val ids = mutableListOf<Long>()

        // Split on word boundaries, preserving punctuation as separate tokens.
        // Pattern: sequences of word-chars (letters, digits, apostrophe) OR a single non-space char.
        val tokenPattern = Regex("[\\w']+|[^\\w\\s]")
        for (match in tokenPattern.findAll(normalized)) {
            val token = match.value

            if (token.all { it.isLetter() || it == '\'' }) {
                // Word — go through the G2P stack.
                val phonemes = wordToArpabet(token.replace("'", ""))
                for (phoneme in phonemes) {
                    val symbol = arpabetToMeloSymbol(phoneme) ?: continue
                    val id = symbolToId[symbol] ?: continue
                    ids.add(id.toLong())
                }
            } else {
                // Punctuation or digit sequence — direct vocab lookup.
                val id = symbolToId[token]
                if (id != null) ids.add(id.toLong())
            }
        }

        return if (ids.isEmpty()) null else ids.toLongArray()
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /**
     * Returns ARPABET phonemes for [word] using the three-tier fallback stack:
     * CMU dict → neural G2P → rule-based letter-by-letter.
     */
    internal fun wordToArpabet(word: String): List<String> {
        if (word.isBlank()) return emptyList()

        // Tier 1: CMU dict.
        cmuDict?.lookup(word)?.let { return it }

        // Tier 2: neural G2P.
        val neural = neuralG2P
        if (neural != null && neural.isLoaded) {
            val result = neural.predict(word)
            if (result.isNotEmpty()) return result
        }

        // Tier 3: rule-based (replicates G2PConverter.ruleBasedG2P).
        return ruleBasedG2P(word.uppercase())
    }

    /**
     * Rule-based grapheme-to-phoneme fallback.
     * Handles common English digraphs and single-letter mappings to ARPABET.
     * Logic mirrors [com.hereliesaz.liperty.ml.G2PConverter.ruleBasedG2P].
     */
    internal fun ruleBasedG2P(word: String): List<String> {
        val phonemes = mutableListOf<String>()
        var i = 0
        while (i < word.length) {
            // Two-character digraph rules.
            if (i < word.length - 1) {
                when (word.substring(i, i + 2)) {
                    "CH" -> { phonemes.add("CH"); i += 2; continue }
                    "SH" -> { phonemes.add("SH"); i += 2; continue }
                    "TH" -> { phonemes.add("TH"); i += 2; continue }
                    "PH" -> { phonemes.add("F"); i += 2; continue }
                    "QU" -> { phonemes.add("K"); phonemes.add("W"); i += 2; continue }
                    "NG" -> { phonemes.add("NG"); i += 2; continue }
                    "ZH" -> { phonemes.add("ZH"); i += 2; continue }
                    "EE" -> { phonemes.add("IY"); i += 2; continue }
                    "OO" -> { phonemes.add("UW"); i += 2; continue }
                }
            }
            // Single-character rules.
            val p: String = when (word[i]) {
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

    /**
     * Maps an ARPABET phoneme (possibly with stress digit, e.g. "AE1") to the
     * MeloTTS symbol string used in [symbolToId].
     *
     * Tries the full form first (stress-marked), then the bare phoneme (no digit).
     * Returns `null` if neither is in the [arpabetToSymbol] map.
     */
    private fun arpabetToMeloSymbol(arpabet: String): String? {
        // Direct match (e.g. "AE1" → some IPA-ish symbol).
        arpabetToSymbol[arpabet]?.let { return it }
        // Strip trailing stress digit and try again (e.g. "AE" → symbol).
        val bare = arpabet.trimEnd { it.isDigit() }
        return arpabetToSymbol[bare]
    }
}
