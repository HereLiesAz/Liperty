package com.hereliesaz.liperty.personalization

import android.content.Context
import org.json.JSONObject

/**
 * A viseme confusion set: a group of words that map to the same viseme
 * sequence under Liperty's fine-grained 12-class viseme map (see
 * `app/src/main/assets/viseme_map_fine.txt`) but to different ARPABET
 * phoneme sequences.
 *
 * From outside the mouth these words look identical — the model cannot
 * distinguish them via coarse lip shape. They're the exact targets of
 * per-user LoRA training: the user records themselves saying each
 * member, and the trained delta learns user-specific micro-features
 * (degree of jaw drop, tongue visibility through teeth gap, asymmetric
 * lip motion, breath plosion, ...) that DO discriminate them.
 *
 * @property visemeSeq Space-separated viseme codes (e.g. "V2 V1A V4").
 *   Identifies the bucket; same for all [words].
 * @property words The confusable words, sorted by corpus frequency
 *   descending (most-frequent first).
 * @property score Mean log-inverse-rank across [words]; higher means
 *   the set is composed of more-frequent words. Used to prioritize
 *   the calibration session — top-N most useful sets first.
 */
data class VisemeConfusionSet(
    val visemeSeq: String,
    val words: List<String>,
    val score: Float,
) {
    /** Number of distinct words in this set. Single-word sets are
     *  pruned at build time (nothing to confuse), so this is >= 2. */
    val size: Int get() = words.size
}

/**
 * Loads the prebuilt `viseme_confusion_sets.json` from the app's
 * assets. The file is produced offline by
 * `tools/build_viseme_confusion_sets.py` and shipped in the APK.
 *
 * The file format is intentionally simple — a `metadata` block and a
 * `sets` array — so the loader is just a JSON parser. No streaming
 * needed; the asset is ~8 KB.
 */
object ConfusionSetLoader {

    private const val ASSET = "viseme_confusion_sets.json"

    /**
     * Read [ASSET] from the context's assets and return the parsed
     * sets in score-descending order (already pre-sorted by the
     * build script; we preserve order rather than re-sorting so the
     * "top-N" UX is deterministic across runs).
     *
     * @param maxSets Optional cap; only the first N sets are returned.
     *   Useful for time-boxing the calibration session — 50 sets ×
     *   ~3 words ≈ 150 utterances ≈ 5-15 minutes.
     * @param maxSetSize Optional cap on words per set, applied AFTER
     *   the build-time `MAX_WORDS_PER_SET`. Lets the UI shorten very
     *   long sets for the calibration flow without rebuilding the
     *   asset.
     */
    fun load(
        context: Context,
        maxSets: Int = Int.MAX_VALUE,
        maxSetSize: Int = Int.MAX_VALUE,
    ): List<VisemeConfusionSet> {
        val text = context.assets.open(ASSET).bufferedReader().use { it.readText() }
        return parse(text, maxSets, maxSetSize)
    }

    /** Pure-string entry point for unit testing. */
    fun parse(
        json: String,
        maxSets: Int = Int.MAX_VALUE,
        maxSetSize: Int = Int.MAX_VALUE,
    ): List<VisemeConfusionSet> {
        val root = JSONObject(json)
        val arr = root.getJSONArray("sets")
        val out = ArrayList<VisemeConfusionSet>()
        val n = minOf(arr.length(), maxSets)
        for (i in 0 until n) {
            val obj = arr.getJSONObject(i)
            val visemeSeq = obj.getString("viseme_seq")
            val wordsArr = obj.getJSONArray("words")
            val words = ArrayList<String>(wordsArr.length())
            val wordCap = minOf(wordsArr.length(), maxSetSize)
            for (w in 0 until wordCap) {
                words.add(wordsArr.getString(w))
            }
            if (words.size < 2) continue   // Defensive — build script already filters
            val score = obj.optDouble("score", 0.0).toFloat()
            out.add(VisemeConfusionSet(visemeSeq, words, score))
        }
        return out
    }
}
