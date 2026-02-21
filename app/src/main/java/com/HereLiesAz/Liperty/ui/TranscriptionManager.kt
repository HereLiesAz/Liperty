package com.HereLiesAz.Liperty.ui

import com.HereLiesAz.Liperty.ml.HomopheneCorrector

class TranscriptionManager {

    private val homopheneCorrector = HomopheneCorrector()

    // List of words in the current sentence
    private val words = mutableListOf<String>()

    // Index of the currently selected word (for editing/swapping)
    private var selectedWordIndex = -1

    fun appendText(text: String) {
        if (text.isEmpty()) return

        // Simple splitting by space. In real VSR, we might get tokens.
        val newWords = text.trim().split("\\s+".toRegex())
        words.addAll(newWords)

        // Select the last word by default
        if (words.isNotEmpty()) {
            selectedWordIndex = words.size - 1
        }
    }

    fun getCurrentSentence(): String {
        return words.joinToString(" ")
    }

    fun clear() {
        words.clear()
        selectedWordIndex = -1
    }

    fun cycleCurrentWord(direction: Int) {
        if (selectedWordIndex == -1 || selectedWordIndex >= words.size) return

        val currentWord = words[selectedWordIndex]
        val alternatives = homopheneCorrector.getAlternatives(currentWord).toMutableList()

        // Add current word to list if not present, to allow cycling back
        if (!alternatives.contains(currentWord.lowercase()) && !alternatives.contains(currentWord)) {
            alternatives.add(0, currentWord)
        } else {
            // Ensure current word is in the list for index finding
             if (!alternatives.contains(currentWord)) {
                 alternatives.add(currentWord)
             }
        }

        // Find current index in alternatives
        // We use case-insensitive comparison for finding, but preserve case if possible?
        // For simplicity, let's work with lowercase or whatever getAlternatives returns.
        val currentIndex = alternatives.indexOfFirst { it.equals(currentWord, ignoreCase = true) }

        if (currentIndex != -1 && alternatives.isNotEmpty()) {
            var newIndex = (currentIndex + direction) % alternatives.size
            if (newIndex < 0) newIndex += alternatives.size

            words[selectedWordIndex] = alternatives[newIndex]
        }
    }

    fun selectWord(index: Int) {
        if (index in 0 until words.size) {
            selectedWordIndex = index
        }
    }

    fun getSelectedWordIndex(): Int {
        return selectedWordIndex
    }
}
