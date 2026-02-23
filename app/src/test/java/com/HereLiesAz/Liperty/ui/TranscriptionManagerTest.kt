package com.HereLiesAz.Liperty.ui

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class TranscriptionManagerTest {

    private lateinit var context: Context
    private lateinit var manager: TranscriptionManager

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        manager = TranscriptionManager(context)
    }

    @Test
    fun testAppendText() {
        manager.appendText("Hello World")
        assertEquals("Hello World", manager.getCurrentSentence())
        assertEquals(1, manager.getSelectedWordIndex()) // 0=Hello, 1=World
    }

    @Test
    fun testCycleWord() {
        // "mat" -> "bat", "pat" (Based on homophones.json)
        manager.appendText("The cat sat on the mat")

        // "mat" is selected (last word)
        // Cycle +1: mat -> bat
        manager.cycleCurrentWord(1)
        assertEquals("The cat sat on the bat", manager.getCurrentSentence())

        // Cycle +1: bat -> pat (order depends on JSON loading, but generally reliable)
        // With "bat": alternatives [mat, pat].
        // If "bat" not in list, add to start: [bat, mat, pat].
        // Index of "bat" is 0. +1 -> 1 ("mat").

        // Let's re-verify the logic in TranscriptionManager:
        /*
        val alternatives = homopheneCorrector.getAlternatives(currentWord).toMutableList()
        // alternatives for "bat" -> ["mat", "pat"]
        if (!alternatives.contains("bat")) alternatives.add(0, "bat")
        // alternatives -> ["bat", "mat", "pat"]
        currentIndex = 0 ("bat")
        newIndex = (0 + 1) % 3 = 1 -> "mat"
        */

        // So cycling "bat" gives "mat".
        manager.cycleCurrentWord(1)
        assertEquals("The cat sat on the mat", manager.getCurrentSentence())

        // Cycle +1: mat -> bat
        manager.cycleCurrentWord(1)
        assertEquals("The cat sat on the bat", manager.getCurrentSentence())
    }

    @Test
    fun testClear() {
        manager.appendText("Test")
        manager.clear()
        assertEquals("", manager.getCurrentSentence())
        assertEquals(-1, manager.getSelectedWordIndex())
    }
}
