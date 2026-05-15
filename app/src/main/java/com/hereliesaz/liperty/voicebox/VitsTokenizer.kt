package com.hereliesaz.liperty.voicebox

import com.hereliesaz.liperty.ml.G2PConverter
import org.json.JSONObject

/**
 * Tokenizes English text into VITS input_ids using the model's own
 * vocabulary (exported in `pocket_tts_phoneme_map.json` by
 * `tools/export_tts_to_onnx.ipynb`).
 *
 * For phoneme-based models (VITS VCTK): text → ARPABET → IPA → char indices.
 * For character-based models: text → lowercase char indices directly.
 *
 * Handles `add_blank` (interleaved blank tokens between every symbol),
 * which is required by the VITS monotonic alignment architecture.
 *
 * Designed to be testable without ONNX Runtime or Android Context —
 * feed it JSON and text, get back a LongArray.
 */
class VitsTokenizer {

    private var charToId: Map<String, Int> = emptyMap()
    private var blankId: Int = 0
    private var addBlank: Boolean = true
    private var isPhonemeModel: Boolean = false
    private val g2p = G2PConverter()

    /** True once [loadFromJson] succeeds. */
    val isLoaded: Boolean get() = charToId.isNotEmpty()

    /**
     * Parses `pocket_tts_phoneme_map.json` content.
     *
     * Expected keys:
     * - `char_to_id` (object): symbol → index mapping (ground truth from the model)
     * - `add_blank` (bool, default true): whether to insert blank tokens
     * - `is_phoneme` (bool): if true, text is converted to IPA before tokenizing
     * - `blank` (string): the blank symbol name (for looking up its index)
     *
     * Returns true on success.
     */
    fun loadFromJson(jsonString: String): Boolean {
        return try {
            val json = JSONObject(jsonString)
            isPhonemeModel = json.optBoolean("is_phoneme", false)
            addBlank = json.optBoolean("add_blank", true)

            val mapping = json.optJSONObject("char_to_id") ?: return false
            charToId = buildMap {
                val keys = mapping.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    put(key, mapping.getInt(key))
                }
            }

            val blankStr = json.optString("blank", "<blnk>")
            blankId = charToId[blankStr] ?: 0

            charToId.isNotEmpty()
        } catch (_: Exception) {
            false
        }
    }

    /**
     * Tokenizes [text] into the LongArray of input_ids the VITS ONNX expects.
     *
     * For phoneme models: English text → IPA (via [G2PConverter]) → char-level
     * index lookup with blank interleaving.
     * For character models: English text → lowercase → char-level index lookup.
     *
     * Unknown symbols are silently skipped.
     * Returns an empty array if no valid tokens are produced.
     */
    fun tokenize(text: String): LongArray {
        if (!isLoaded) return LongArray(0)

        val processedText = if (isPhonemeModel) textToIpa(text) else text.lowercase()

        val indices = mutableListOf<Long>()
        if (addBlank) indices.add(blankId.toLong())

        for (char in processedText) {
            val id = charToId[char.toString()] ?: continue
            indices.add(id.toLong())
            if (addBlank) indices.add(blankId.toLong())
        }

        return indices.toLongArray()
    }

    /**
     * Converts English text to an IPA string with spaces between words.
     *
     * Uses [G2PConverter] for ARPABET → IPA conversion. Each ARPABET
     * phoneme maps to one or more IPA characters; those are concatenated
     * within a word, and words are separated by a single space.
     */
    internal fun textToIpa(text: String): String {
        val words = text.trim().split(Regex("\\s+"))
        return words.mapNotNull { word ->
            val clean = word.replace(Regex("[^a-zA-Z']"), "")
            if (clean.isEmpty()) return@mapNotNull null
            g2p.wordToPhonemes(clean.uppercase())
                .joinToString("") { g2p.toIPA(it) }
        }.joinToString(" ")
    }
}
