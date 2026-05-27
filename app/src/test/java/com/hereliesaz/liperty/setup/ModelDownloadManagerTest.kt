package com.hereliesaz.liperty.setup

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/**
 * Guards the first-launch download manifest against the regression that caused
 * the post-download force-close: the active SyncVSR seq2seq backend needs the
 * vocab + encoder + decoder, but they were missing from [ModelDownloadManager.models],
 * so they were never downloaded and the pipeline crashed on the unloaded sessions.
 *
 * Uses AndroidJUnit4 + sdk=34 per the project's Robolectric/targetSdk=37 workaround.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class ModelDownloadManagerTest {

    private lateinit var manager: ModelDownloadManager

    @Before
    fun setUp() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        manager = ModelDownloadManager(context)
    }

    @Test
    fun manifestIncludesSeq2SeqModelsAsRequired() {
        val required = manager.models.filter { it.required }.map { it.fileName }.toSet()

        // The active backend (VSR_BACKEND == BACKEND_SYNC_VSR, SYNCVSR_USE_SEQ2SEQ)
        // cannot assemble without these three. They must be fetched at first
        // launch — none of them is bundled in the production APK.
        assertTrue("vocab must be required", "syncvsr_unigram_units.txt" in required)
        assertTrue("seq2seq encoder must be required", "syncvsr_lrs3_encoder.onnx" in required)
        assertTrue("seq2seq decoder must be required", "syncvsr_lrs3_decoder.onnx" in required)

        // The CTC fallback model and face detector remain required too.
        assertTrue("CTC model must be required", "syncvsr_lrs3_visual_ctc_fp16.onnx" in required)
        assertTrue("face landmarker must be required", "face_landmarker.task" in required)
    }

    @Test
    fun seq2SeqModelSpecsPointAtSyncVsrRepo() {
        val byName = manager.models.associateBy { it.fileName }
        for (name in listOf(
            "syncvsr_unigram_units.txt",
            "syncvsr_lrs3_encoder.onnx",
            "syncvsr_lrs3_decoder.onnx",
        )) {
            val spec = byName[name]
            assertTrue("$name present in manifest", spec != null)
            org.junit.Assert.assertEquals(
                "$name must be fetched from the syncvsr repo",
                "HereLiesAz/liperty-syncvsr-onnx", spec!!.repo
            )
        }
    }

    @Test
    fun freshInstallIsNotSetupComplete() {
        // No models on disk -> setup is not complete (the setup screen must show).
        clearPrefs()
        assertFalse(manager.isSetupComplete())
    }

    @Test
    fun manifestVersionBumpForcesRecheckForOldInstalls() {
        // Simulate a pre-existing install that completed setup under an older
        // manifest version (before the seq2seq models were required). The
        // version bump must invalidate the stale setup_complete flag so the
        // new required models get pulled. With no models on disk, the result
        // is "not complete".
        val prefs = ApplicationProvider.getApplicationContext<Context>()
            .getSharedPreferences("ModelDownloadPrefs", Context.MODE_PRIVATE)
        prefs.edit()
            .putBoolean("setup_complete", true)
            .putInt("setup_version", 2)
            .apply()

        assertFalse(
            "stale setup_complete at an older version must not short-circuit",
            manager.isSetupComplete()
        )
    }

    private fun clearPrefs() {
        ApplicationProvider.getApplicationContext<Context>()
            .getSharedPreferences("ModelDownloadPrefs", Context.MODE_PRIVATE)
            .edit().clear().apply()
    }
}
