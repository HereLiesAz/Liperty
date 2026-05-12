package com.hereliesaz.liperty.ml

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/**
 * Unit tests for [KenLmScorer].
 *
 * Uses [AndroidJUnit4] + sdk=34 only because [KenLmScorer] writes
 * diagnostics through `android.util.Log`. The Robolectric runtime stubs
 * Log out so we don't crash with "not mocked" errors; we don't actually
 * exercise any Android-specific behavior here.
 *
 * The native libkenlm.so is not loaded in JVM-only unit tests (no NDK
 * artifact on the test classpath), so these tests exercise the
 * *fallback* behavior: when native loading fails, the scorer must
 * advertise itself as "not native-loaded" and silently return 0 for
 * every score — i.e. degrade to a no-op without crashing the decoder.
 *
 * Integration tests on a real device cover the happy path.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class KenLmScorerTest {

    @Test
    fun reportsNotLoadedWhenNativeLibraryAbsent() {
        val s = KenLmScorer.tryLoad("/nonexistent/path.bin")
        assertFalse("native not available in JVM unit tests", s.isNativeLoaded)
    }

    @Test
    fun scoresReturnZeroWhenNotLoaded() {
        val s = KenLmScorer.tryLoad("/nonexistent/path.bin")
        assertEquals(0f, s.score(emptyList()), 0f)
        assertEquals(0f, s.score(listOf("the", "cat")), 0f)
    }

    @Test
    fun closeIsIdempotentEvenIfNeverInitialized() {
        val s = KenLmScorer.tryLoad("/nonexistent/path.bin")
        s.close()
        s.close()    // second close must not throw
    }

    @Test
    fun scoreAcceptsArbitraryCasingButUppercasesForLibrispeechVocab() {
        // The contract documented on [KenLmScorer]: callers may pass
        // lowercase words because the LibriSpeech LM is uppercase-normalized.
        // Without the native lib, score() still returns 0, but it must not
        // crash on lowercase input.
        val s = KenLmScorer.tryLoad("/nonexistent/path.bin")
        assertEquals(0f, s.score(listOf("hello", "world")), 0f)
        assertEquals(0f, s.score(listOf("HELLO", "WORLD")), 0f)
    }
}
