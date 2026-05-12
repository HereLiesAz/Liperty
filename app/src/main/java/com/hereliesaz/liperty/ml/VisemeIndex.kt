package com.hereliesaz.liperty.ml

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * Word ↔ viseme-sequence lookup used by [VisemeRescorer].
 *
 * The on-disk asset (`viseme_index.json`, produced by
 * `tools/build_viseme_index.py`) is a flat inverse map
 * `{viseme_sequence: [word, word, ...]}`. At load time we walk it once
 * to build a forward map (word → viseme_sequence) so [candidatesFor]
 * can do `word -> seq -> [candidate_words]` in O(1).
 *
 * Memory footprint at the cmudict scale (~126k words / 30k unique
 * viseme sequences): forward map ~5 MB, inverse map ~2 MB. Acceptable
 * for what it buys us — viseme-aware rescoring is the headline feature
 * for visual ASR.
 */
class VisemeIndex(
    /** Inverse map (asset-native): viseme_seq -> sorted candidate words. */
    private val invIndex: Map<String, List<String>>,
) {

    /** Forward map (built at construction): word -> viseme_seq. */
    private val wordToSeq: Map<String, String> = HashMap<String, String>(invIndex.values.sumOf { it.size }).apply {
        for ((seq, words) in invIndex) {
            for (w in words) put(w, seq)
        }
    }

    /**
     * Words that share the input word's viseme sequence — i.e. the set
     * of word substitutions that would be visually consistent with what
     * the encoder saw. Always includes the input word itself if known.
     *
     * Unknown words (not in cmudict, or stripped during build) return
     * `listOf(word)` so the rescorer treats them as un-substitutable.
     *
     * @param maxCandidates cap on the returned set size. The input word
     *   is always included; the rest are filled deterministically (the
     *   index pre-sorts alphabetically). Default 10 keeps total LM calls
     *   bounded for short sentences.
     */
    fun candidatesFor(word: String, maxCandidates: Int = 10): List<String> {
        val key = word.uppercase()
        val seq = wordToSeq[key] ?: return listOf(key)
        val all = invIndex[seq] ?: return listOf(key)
        if (all.size <= maxCandidates) return all
        // Cap: always keep the input word, fill the rest from the head
        // of the (alphabetically sorted) list.
        val kept = ArrayList<String>(maxCandidates)
        kept.add(key)
        for (w in all) {
            if (kept.size >= maxCandidates) break
            if (w != key) kept.add(w)
        }
        return kept
    }

    companion object {
        fun fromJson(text: String): VisemeIndex {
            val obj = JSONObject(text)
            val map = HashMap<String, List<String>>(obj.length())
            val keys = obj.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                val arr: JSONArray = obj.getJSONArray(k)
                val list = ArrayList<String>(arr.length())
                for (i in 0 until arr.length()) list.add(arr.getString(i))
                map[k] = list
            }
            return VisemeIndex(map)
        }

        fun loadFromAssets(context: Context, assetName: String = "viseme_index.json"): VisemeIndex =
            context.assets.open(assetName).bufferedReader().use { fromJson(it.readText()) }
    }
}
