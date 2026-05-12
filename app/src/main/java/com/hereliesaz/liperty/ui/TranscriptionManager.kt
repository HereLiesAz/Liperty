package com.hereliesaz.liperty.ui

import android.content.Context
import com.hereliesaz.liperty.ml.HomopheneCorrector
import com.hereliesaz.liperty.ml.LanguageModel

/**
 * Manages the list of transcribed words, their confidences, and autocorrection logic.
 *
 * @param transformSentence Optional sentence-level post-processor invoked by
 *   [getCleanedSentence] over the assembled transcript. Auto-AVSR backends
 *   wire this to a synchronous wrapper around `LlmTextCleaner.clean()` to
 *   apply a small LLM cleanup pass on top of the existing word-level
 *   homophone+LM corrections. The result is cached and only re-evaluated
 *   when the underlying transcript changes (appendText / clear / cycle).
 */
class TranscriptionManager(
    private val context: Context,
    private val transformSentence: ((String) -> String)? = null,
) {

    companion object {
        private val WHITESPACE_REGEX = "\\s+".toRegex()
    }

    private val homopheneCorrector = HomopheneCorrector(context)
    private val languageModel = LanguageModel()

    // Each entry is (word, confidence) — confidence comes from the VSR model's softmax output.
    // wordEntries = committed words (stable; appeared in a previous inference window
    //               that the next window confirmed via overlap dedupe).
    // liveEntries = the latest window's "new tail" — words decoded from the most
    //               recent frames that haven't yet been confirmed by a subsequent
    //               window. Rendered dimmer in the UI as a live preview.
    private val wordEntries = mutableListOf<Pair<String, Float>>()
    private val liveEntries = mutableListOf<Pair<String, Float>>()
    private var selectedWordIndex = -1

    /** Cache of (signature, transformed-sentence) so the transformer doesn't
     *  re-run on every read. The signature is a cheap stand-in for the
     *  transcript content — content size + the assembled string itself —
     *  invalidated automatically by every appendText / clear / cycle. */
    private var cachedTransformed: Pair<String, String>? = null

    /**
     * Streaming append from an inference window.
     *
     * The VSR pipeline runs on a 16-frame sliding window that retains 8 frames
     * of overlap with the previous window (see MainActivity.AUTOAVSR_SLIDE_RETAIN).
     * So consecutive inferences naturally share their first few decoded words.
     * Without dedupe, the visible transcript would repeat — "the quick brown the
     * quick brown fox jumps the quick brown fox jumps over". We:
     *
     *   1. Promote whatever was "live" from the previous call to committed
     *      (the new window confirmed those words by including them in its prefix).
     *   2. Find the longest suffix of the now-committed transcript that matches
     *      a prefix of the new window's words, and drop that prefix.
     *   3. Whatever remains becomes the new live preview.
     *
     * Homophone + LM correction runs over the live tail before storage so the
     * UI shows already-corrected words rather than flickering between hypotheses.
     *
     * @param confidence Mean max-softmax probability for this inference window (0–1).
     */
    @Synchronized
    fun appendText(text: String, confidence: Float = 0f) {
        if (text.isEmpty()) {
            // Empty window: don't clear live -- a transient blank between
            // two real inferences shouldn't wipe the user's preview.
            return
        }
        val newWords = text.trim().split(WHITESPACE_REGEX).filter { it.isNotEmpty() }
        if (newWords.isEmpty()) return

        // 1. Promote previous live words to committed. They survived to the next
        //    inference, which is the strongest stability signal we have.
        if (liveEntries.isNotEmpty()) {
            wordEntries.addAll(liveEntries)
            liveEntries.clear()
        }

        // 2. Suffix-prefix dedupe against the committed transcript so we only
        //    append the new tail (words decoded from the *new* 8 frames in this
        //    window, not the 8 overlap frames).
        val committedTokens = wordEntries.map { it.first }
        val overlapK = longestSuffixPrefixOverlap(committedTokens, newWords)
        val newTail = newWords.drop(overlapK)

        if (newTail.isEmpty()) return  // window was entirely overlap

        // 3. Homophone+LM correction on the new tail, using the rolling "previous
        //    word" context. Same logic as the pre-streaming version, just scoped
        //    to the tail instead of the whole window output.
        var prevWord = wordEntries.lastOrNull()?.first
        for (word in newTail) {
            var bestWord = word
            if (prevWord != null) {
                val alternatives = buildAlternatives(word)
                if (alternatives.size > 1) {
                    var bestScore = -1.0
                    val normalizedPrev = prevWord.lowercase().replace(Regex("[^a-z']"), "")
                    for (alt in alternatives) {
                        val score = languageModel.getWordScore(normalizedPrev, alt)
                        if (score > bestScore) {
                            bestScore = score
                            bestWord = alt
                        }
                    }
                }
            }
            val formattedWord = applyOriginalCasing(word, bestWord)
            liveEntries.add(formattedWord to confidence)
            prevWord = formattedWord
        }

        // Selection points into the combined (committed + live) list so the
        // user can keep tapping/cycling the most recent word regardless of
        // which buffer it lives in. cycleCurrentWord routes the mutation to
        // the right buffer based on the index.
        val total = wordEntries.size + liveEntries.size
        if (total > 0) selectedWordIndex = total - 1
    }

    /**
     * Longest k such that the last k tokens of `existing` match the first k
     * of `incoming` (case-insensitive). Used by streaming dedupe so the
     * sliding window's overlap region doesn't double-emit words.
     */
    private fun longestSuffixPrefixOverlap(existing: List<String>, incoming: List<String>): Int {
        val maxK = minOf(existing.size, incoming.size)
        for (k in maxK downTo 1) {
            var match = true
            for (i in 0 until k) {
                if (!existing[existing.size - k + i].equals(incoming[i], ignoreCase = true)) {
                    match = false
                    break
                }
            }
            if (match) return k
        }
        return 0
    }

    private fun buildAlternatives(word: String): List<String> {
        val alternatives = homopheneCorrector.getAlternatives(word).toMutableList()
        if (!alternatives.contains(word.lowercase()) && !alternatives.contains(word)) {
            alternatives.add(0, word)
        } else if (!alternatives.contains(word)) {
            alternatives.add(word)
        }
        return alternatives
    }

    private fun applyOriginalCasing(original: String, corrected: String): String {
        if (original.isEmpty()) return corrected

        return when {
            // Preserve acronyms / all-uppercase tokens, e.g. "USA" -> "USE"
            original.all { it.isUpperCase() } -> corrected.uppercase()

            // Preserve leading capital for capitalized words, e.g. "London" -> "Paris"
            original.first().isUpperCase() ->
                corrected.replaceFirstChar { ch ->
                    if (ch.isLowerCase()) ch.titlecase() else ch.toString()
                }

            // Default: keep corrected as-is
            else -> corrected
        }
    }

    /** Visible transcript: committed words plus the live preview tail.
     *  Callers wanting only stable words should use [getCommittedWords]. */
    @Synchronized
    fun getCurrentSentence(): String =
        (wordEntries.asSequence() + liveEntries.asSequence())
            .joinToString(" ") { it.first }

    /**
     * Returns the assembled sentence after the optional [transformSentence]
     * post-processor. When no transformer was supplied, returns
     * [getCurrentSentence] unchanged. The transformed result is cached and
     * only re-evaluated when the underlying assembled sentence changes.
     */
    @Synchronized
    fun getCleanedSentence(): String {
        val current = (wordEntries.asSequence() + liveEntries.asSequence())
            .joinToString(" ") { it.first }
        val transformer = transformSentence ?: return current

        val cached = cachedTransformed
        if (cached != null && cached.first == current) return cached.second

        val transformed = transformer(current)
        cachedTransformed = current to transformed
        return transformed
    }

    @Synchronized
    fun clear() {
        wordEntries.clear()
        liveEntries.clear()
        selectedWordIndex = -1
    }

    /** Committed words only — the stable transcript that the UI renders
     *  fully bright. Excludes the latest window's live preview tail. */
    @Synchronized
    fun getCommittedWords(): List<String> = wordEntries.map { it.first }

    @Synchronized
    fun getCommittedConfidences(): List<Float> = wordEntries.map { it.second }

    /** Live-preview words: latest window's new tail, not yet committed.
     *  Rendered with reduced alpha / italic in the UI so the user can
     *  see speech *as it's mouthed* before the next window confirms it. */
    @Synchronized
    fun getLiveWords(): List<String> = liveEntries.map { it.first }

    @Synchronized
    fun getLiveConfidences(): List<Float> = liveEntries.map { it.second }

    @Synchronized
    fun cycleCurrentWord(direction: Int) {
        val total = wordEntries.size + liveEntries.size
        if (selectedWordIndex !in 0 until total) return

        // Route to whichever buffer holds the selected word.
        val (buffer, localIdx) = if (selectedWordIndex < wordEntries.size)
            wordEntries to selectedWordIndex
        else
            liveEntries to (selectedWordIndex - wordEntries.size)

        val currentWord = buffer[localIdx].first
        val alternatives = buildAlternatives(currentWord)

        val currentIndex = alternatives.indexOfFirst { it.equals(currentWord, ignoreCase = true) }
        if (currentIndex != -1 && alternatives.isNotEmpty()) {
            var newIndex = (currentIndex + direction) % alternatives.size
            if (newIndex < 0) newIndex += alternatives.size
            val conf = buffer[localIdx].second
            buffer[localIdx] = Pair(alternatives[newIndex], conf)
        }
    }

    @Synchronized
    fun selectWord(index: Int) {
        val total = wordEntries.size + liveEntries.size
        if (index in 0 until total) selectedWordIndex = index
    }

    @Synchronized
    fun getSelectedWordIndex(): Int = selectedWordIndex

    /** All visible words (committed + live), matching [getCurrentSentence]. */
    @Synchronized
    fun getWords(): List<String> =
        (wordEntries.asSequence() + liveEntries.asSequence()).map { it.first }.toList()

    @Synchronized
    fun getWordConfidences(): List<Float> =
        (wordEntries.asSequence() + liveEntries.asSequence()).map { it.second }.toList()
}
