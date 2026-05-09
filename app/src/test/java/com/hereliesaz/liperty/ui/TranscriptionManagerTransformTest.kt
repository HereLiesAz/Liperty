package com.hereliesaz.liperty.ui

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/**
 * Tests for [TranscriptionManager.getCleanedSentence] — the optional
 * sentence-level transformer hook used by the Auto-AVSR backend to wire
 * `LlmTextCleaner` (or any other text post-processor) as a final pass over
 * the assembled transcript.
 *
 * Uses AndroidJUnit4 + @Config(sdk=[34]) because Robolectric on the
 * project's targetSdk=37 fails to load (same issue as the legacy
 * TranscriptionManagerTest).
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class TranscriptionManagerTransformTest {

    private lateinit var context: Context

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
    }

    @Test
    fun withoutTransformerCleanedEqualsCurrent() {
        val mgr = TranscriptionManager(context)
        mgr.appendText("hello world")
        assertEquals(mgr.getCurrentSentence(), mgr.getCleanedSentence())
    }

    @Test
    fun withTransformerAppliesItToAssembledSentence() {
        val mgr = TranscriptionManager(context, transformSentence = { it.uppercase() })
        mgr.appendText("hello world")
        assertEquals("HELLO WORLD", mgr.getCleanedSentence())
    }

    @Test
    fun transformerIsCachedAndRunsOncePerContent() {
        var calls = 0
        val mgr = TranscriptionManager(context, transformSentence = {
            calls++
            "$it (cleaned)"
        })
        mgr.appendText("foo")
        mgr.getCleanedSentence()
        mgr.getCleanedSentence()
        mgr.getCleanedSentence()
        assertEquals("transformer should run once for unchanged text", 1, calls)
    }

    @Test
    fun transformerReRunsAfterAppendText() {
        var calls = 0
        val mgr = TranscriptionManager(context, transformSentence = {
            calls++
            it
        })
        mgr.appendText("foo")
        mgr.getCleanedSentence()
        mgr.appendText("bar")
        mgr.getCleanedSentence()
        assertEquals(2, calls)
    }

    @Test
    fun transformerReRunsAfterClear() {
        var calls = 0
        val mgr = TranscriptionManager(context, transformSentence = {
            calls++
            it
        })
        mgr.appendText("foo")
        mgr.getCleanedSentence()
        mgr.clear()
        mgr.appendText("bar")
        mgr.getCleanedSentence()
        assertEquals(2, calls)
    }

    @Test
    fun transformerReceivesTheCorrectedAssembledSentence() {
        // The transformer should receive the post-homophone-correction
        // sentence — i.e. whatever getCurrentSentence() would return —
        // not the raw input. We verify by checking the input length or
        // a property the transformer can check.
        var seen: String? = null
        val mgr = TranscriptionManager(context, transformSentence = {
            seen = it
            it
        })
        mgr.appendText("hello world")
        mgr.getCleanedSentence()
        assertEquals(mgr.getCurrentSentence(), seen)
    }

    @Test
    fun transformerCanReturnDifferentLengthOutput() {
        // LLM cleaners may add or drop tokens. Verify TranscriptionManager
        // accepts arbitrary output strings, not just same-length transforms.
        val mgr = TranscriptionManager(context, transformSentence = { "the cat sat on the mat" })
        mgr.appendText("th c sa")
        assertEquals("the cat sat on the mat", mgr.getCleanedSentence())
    }
}
