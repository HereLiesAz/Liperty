package com.hereliesaz.liperty.ml

import android.content.Context

/**
 * Parses the `viseme_map.txt` asset and exposes phoneme→viseme lookups
 * for [VisemeRescorer]. See [VisemeMapTest] and the asset file itself
 * for the format and the linguistic rationale behind the groupings.
 *
 * The map is small (~40 phonemes, 9 viseme classes) so we hold it as a
 * plain `Map<String, String>` in memory — no need for tries or other
 * optimization.
 */
object VisemeMap {

    /** Pure parse step. Strips `#`-comments and blank lines; expects
     *  one `<PHONEME>\t<VISEME>` per data line (whitespace tolerant). */
    fun parse(text: String): Map<String, String> {
        val out = LinkedHashMap<String, String>()
        for (raw in text.lineSequence()) {
            val line = raw.substringBefore('#').trim()
            if (line.isEmpty()) continue
            val parts = line.split(Regex("\\s+"), limit = 2)
            if (parts.size != 2) continue
            val phoneme = parts[0]
            val viseme = parts[1].trim()
            if (phoneme.isEmpty() || viseme.isEmpty()) continue
            out[phoneme] = viseme
        }
        return out
    }

    fun loadFromAssets(context: Context, assetName: String = "viseme_map.txt"): Map<String, String> =
        context.assets.open(assetName).bufferedReader().use { parse(it.readText()) }

    /** Convert a sequence of ARPABET phonemes to viseme codes. Unknown
     *  phonemes are silently dropped (see [VisemeMapTest] for rationale). */
    fun phonemesToVisemes(phonemes: List<String>, map: Map<String, String>): List<String> =
        phonemes.mapNotNull { map[it] }
}
