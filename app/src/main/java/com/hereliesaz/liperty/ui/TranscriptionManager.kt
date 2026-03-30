package com.hereliesaz.liperty.ui

import android.content.Context
import com.hereliesaz.liperty.ml.HomopheneCorrector
import com.hereliesaz.liperty.ml.LanguageModel

class TranscriptionManager(private val context: Context) {

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
        val newWords = text.trim().split("\\s+".toRegex())

        for (word in newWords) {
            var bestWord = word
            val prevWord = wordEntries.lastOrNull()?.first

            if (prevWord != null) {
                // Get homophenes
                val alternatives = homopheneCorrector.getAlternatives(word).toMutableList()
                if (!alternatives.contains(word.lowercase()) && !alternatives.contains(word)) {
                    alternatives.add(0, word)
                } else if (!alternatives.contains(word)) {
                    alternatives.add(word)
                }

                // Score them with the language model if there are alternatives
                if (alternatives.size > 1) {
                    var bestScore = -1.0
                    for (alt in alternatives) {
                        val score = languageModel.getWordScore(prevWord, alt)
                        if (score > bestScore) {
                            bestScore = score
                            bestWord = alt
                        }
                    }
                }
            }

            // Try to match original casing if we changed it, assuming the model output is lower case or capitalized
            val formattedWord = if (word.firstOrNull()?.isUpperCase() == true) {
                bestWord.replaceFirstChar { it.uppercase() }
            } else {
                bestWord
            }

            wordEntries.add(Pair(formattedWord, confidence))
        }

        if (wordEntries.isNotEmpty()) selectedWordIndex = wordEntries.size - 1
    }

    fun getCurrentSentence(): String = wordEntries.joinToString(" ") { it.first }

    fun clear() {
        wordEntries.clear()
        selectedWordIndex = -1
    }

    fun cycleCurrentWord(direction: Int) {
        if (selectedWordIndex == -1 || selectedWordIndex >= wordEntries.size) return

        val currentWord = wordEntries[selectedWordIndex].first
        val alternatives = homopheneCorrector.getAlternatives(currentWord).toMutableList()

        if (!alternatives.contains(currentWord.lowercase()) && !alternatives.contains(currentWord)) {
            alternatives.add(0, currentWord)
        } else {
            if (!alternatives.contains(currentWord)) alternatives.add(currentWord)
        }

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
