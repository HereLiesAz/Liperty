package com.hereliesaz.liperty.ml

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * On-device instrumented tests for [KenLmScorer].
 *
 * These tests validate the full JNI path: `libliperty_cv.so` →
 * `kenlm_jni.cpp` (with `KENLM_AVAILABLE=1`) → real `lm::ngram::Model`
 * scoring against `librispeech_3gram.bin`.
 *
 * **Requires:** arm64-v8a device with the KenLM model bundled in assets
 * (placed there by `setup_libs.sh`).
 *
 * Run with:
 * ```
 * ./gradlew connectedDebugAndroidTest --tests "*.KenLmScorerDeviceTest"
 * ```
 */
@RunWith(AndroidJUnit4::class)
class KenLmScorerDeviceTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private lateinit var modelFile: File
    private var scorer: KenLmScorer? = null

    @Before
    fun copyModelFromAssets() {
        modelFile = File(context.filesDir, "librispeech_3gram.bin")
        if (!modelFile.exists() || modelFile.length() < 1000) {
            context.assets.open("librispeech_3gram.bin").use { input ->
                modelFile.outputStream().use { output -> input.copyTo(output) }
            }
        }
    }

    @After
    fun cleanup() {
        scorer?.close()
    }

    @Test
    fun modelFileIsCopiedFromAssets() {
        assertTrue("Model file should exist after copy", modelFile.exists())
        assertTrue(
            "Model file should be ~27 MB (got ${modelFile.length()} bytes)",
            modelFile.length() > 20_000_000
        )
    }

    @Test
    fun nativeLibraryLoads() {
        scorer = KenLmScorer.tryLoad(modelFile.absolutePath)
        assertTrue(
            "isNativeLoaded must be true on arm64 device with prebuilt libkenlm.a",
            scorer!!.isNativeLoaded
        )
    }

    @Test
    fun scoresNonZeroForEnglishSentence() {
        scorer = KenLmScorer.tryLoad(modelFile.absolutePath)
        // KenLM returns log10 probabilities, which are negative for real
        // sentences. A non-zero score confirms the native bridge is working
        // end-to-end.
        val score = scorer!!.score(listOf("THE", "CAT", "SAT", "ON", "THE", "MAT"))
        assertNotEquals("Score for common English phrase must be non-zero", 0f, score)
        assertTrue("Log10 probability should be negative (got $score)", score < 0f)
    }

    @Test
    fun scoresZeroForEmptyInput() {
        scorer = KenLmScorer.tryLoad(modelFile.absolutePath)
        assertEquals(0f, scorer!!.score(emptyList()), 0f)
    }

    @Test
    fun closeDoesNotCrash() {
        scorer = KenLmScorer.tryLoad(modelFile.absolutePath)
        scorer!!.score(listOf("HELLO", "WORLD"))
        scorer!!.close()
        scorer = null // prevent double-close in @After
    }
}
