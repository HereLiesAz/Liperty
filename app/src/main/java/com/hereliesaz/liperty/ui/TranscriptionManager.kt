package com.hereliesaz.liperty.ui

import android.content.Context
import com.hereliesaz.liperty.ml.HomopheneCorrector
import com.hereliesaz.liperty.ml.LanguageModel

/**
 * Manages the list of transcribed words, their confidences, and autocorrection logic.
 */
class TranscriptionManager(private val context: Context) {

    companion object {
        private val WHITESPACE_REGEX = "\\s+".toRegex()
    }

    private val homopheneCorrector = HomopheneCorrector(context)
    private val languageModel = LanguageModel()

    // Each entry is (word, confidence) — confidence comes from the VSR model's softmax output.
    private val wordEntries = mutableListOf<Pair<String, Float>>()
    private var selectedWordIndex = -1

    /**
     * Appends new words from an inference result.
     * Uses a language model to automatically correct common homophenes based on context.
     * @param confidence Mean max-softmax probability for this inference window (0–1).
     */
    fun appendText(text: String, confidence: Float = 0f) {
        if (text.isEmpty()) return
        val newWords = text.trim().split(WHITESPACE_REGEX)

        for (word in newWords) {
            var bestWord = word
            val prevWord = wordEntries.lastOrNull()?.first

            if (prevWord != null) {
                val alternatives = buildAlternatives(word)

                // Score them with the language model if there are alternatives
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
            wordEntries.add(Pair(formattedWord, confidence))
        }

        if (wordEntries.isNotEmpty()) selectedWordIndex = wordEntries.size - 1
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

    fun getCurrentSentence(): String = wordEntries.joinToString(" ") { it.first }

    fun clear() {
        wordEntries.clear()
        selectedWordIndex = -1
    }

    fun cycleCurrentWord(direction: Int) {
        if (selectedWordIndex == -1 || selectedWordIndex >= wordEntries.size) return

        val currentWord = wordEntries[selectedWordIndex].first
        val alternatives = buildAlternatives(currentWord)

        val currentIndex = alternatives.indexOfFirst { it.equals(currentWord, ignoreCase = true) }
        if (currentIndex != -1 && alternatives.isNotEmpty()) {
            var newIndex = (currentIndex + direction) % alternatives.size
            if (newIndex < 0) newIndex += alternatives.size
            val conf = wordEntries[selectedWordIndex].second
            wordEntries[selectedWordIndex] = Pair(alternatives[newIndex], conf)
        }
    }

    fun selectWord(index: Int) {
        if (index in 0 until wordEntries.size) selectedWordIndex = index
    }

    fun getSelectedWordIndex(): Int = selectedWordIndex

    fun getWords(): List<String> = wordEntries.map { it.first }

    fun getWordConfidences(): List<Float> = wordEntries.map { it.second }
}
