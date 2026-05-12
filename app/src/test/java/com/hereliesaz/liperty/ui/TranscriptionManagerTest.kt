package com.hereliesaz.liperty.ui

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

// SDK 37 workaround: targetSdk=37 confuses Robolectric. Pinning to SDK 34
// here matches the pattern documented in CLAUDE.md (used by
// TranscriptionManagerTransformTest, LlmTextCleanerTest, etc.).
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
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
        // Cycle +1: mat -> but
        manager.cycleCurrentWord(1)
        assertEquals("The cat sat on the but", manager.getCurrentSentence())

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

        // Cycle +1: mat -> but
        manager.cycleCurrentWord(1)
        assertEquals("The cat sat on the but", manager.getCurrentSentence())
    }

    @Test
    fun testClear() {
        manager.appendText("Test")
        manager.clear()
        assertEquals("", manager.getCurrentSentence())
        assertEquals(-1, manager.getSelectedWordIndex())
    }

    // Streaming sliding-window dedupe: the VSR pipeline retains 8 of 16
    // frames between inferences, so consecutive windows share prefix words.
    // Without dedupe the visible transcript would repeat itself.
    @Test
    fun testStreamingOverlapDedupe() {
        manager.appendText("the quick brown")
        // First window's words are live, awaiting confirmation.
        assertEquals(emptyList<String>(), manager.getCommittedWords())
        assertEquals(listOf("the", "quick", "brown"), manager.getLiveWords())

        // Second window overlaps by 2 words.
        manager.appendText("quick brown fox")
        // The first window's words got promoted to committed.
        assertEquals(listOf("the", "quick", "brown"), manager.getCommittedWords())
        // Only the new tail ("fox") is live.
        assertEquals(listOf("fox"), manager.getLiveWords())

        // Combined visible transcript shouldn't duplicate.
        assertEquals("the quick brown fox", manager.getCurrentSentence())
    }

    @Test
    fun testStreamingFullOverlapEmitsNothing() {
        manager.appendText("hello world")
        manager.appendText("hello world") // identical window output
        // "hello world" got promoted to committed by the second call; the
        // second call's full content was overlap so nothing new is live.
        assertEquals(listOf("hello", "world"), manager.getCommittedWords())
        assertEquals(emptyList<String>(), manager.getLiveWords())
        assertEquals("hello world", manager.getCurrentSentence())
    }

    @Test
    fun testStreamingDisjointWindowsAppend() {
        manager.appendText("first chunk")
        manager.appendText("unrelated tail") // no overlap at all
        // First chunk promoted, second chunk is the new live tail.
        assertEquals(listOf("first", "chunk"), manager.getCommittedWords())
        assertEquals(listOf("unrelated", "tail"), manager.getLiveWords())
    }

    @Test
    fun testStreamingCaseInsensitiveDedupe() {
        manager.appendText("HELLO World")
        manager.appendText("hello world fox")
        assertEquals(listOf("HELLO", "World"), manager.getCommittedWords())
        assertEquals(listOf("fox"), manager.getLiveWords())
    }
}
