package com.hereliesaz.liperty.ui

import android.content.Context
import com.hereliesaz.liperty.ml.HomopheneCorrector

class TranscriptionManager(private val context: Context) {

    private val homopheneCorrector = HomopheneCorrector(context)

    // Each entry is (word, confidence) — confidence comes from the VSR model's softmax output.
    private val wordEntries = mutableListOf<Pair<String, Float>>()
    private var selectedWordIndex = -1

    /**
     * Appends new words from an inference result.
     * @param confidence Mean max-softmax probability for this inference window (0–1).
     */
    fun appendText(text: String, confidence: Float = 0f) {
        if (text.isEmpty()) return
        val newWords = text.trim().split("\\s+".toRegex())
        newWords.forEach { wordEntries.add(Pair(it, confidence)) }
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
