package com.hereliesaz.liperty.setup

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowNetworkCapabilities

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
        // Clear shared state so tests don't leak setup_complete/version across runs.
        context.getSharedPreferences("ModelDownloadPrefs", Context.MODE_PRIVATE)
            .edit().clear().apply()
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

    @Test
    fun wifiOnlyDefaultsToTrue() {
        assertTrue("Wi-Fi-only must default on to protect battery/data", manager.isWifiOnly())
    }

    @Test
    fun setWifiOnlyPersists() = runBlocking {
        manager.setWifiOnly(false)
        assertFalse(manager.isWifiOnly())
        manager.setWifiOnly(true)
        assertTrue(manager.isWifiOnly())
    }

    @Test
    fun cellularWithWifiOnlyBlocksDownload() = runBlocking {
        // Active network is cellular; Wi-Fi-only is on (default).
        setActiveTransport(NetworkCapabilities.TRANSPORT_CELLULAR)

        manager.downloadAll(skipOptional = false)

        assertTrue(
            "cellular + Wi-Fi-only must block, not download",
            manager.state.value.blockedOnNetwork
        )
        // The first required model should be parked waiting for Wi-Fi, never ERROR.
        assertEquals(
            ModelDownloadManager.Status.WAITING_WIFI,
            manager.state.value.modelStates["face_landmarker.task"]?.status
        )
    }

    /** Points the active network's capabilities at a single transport type. */
    private fun setActiveTransport(transport: Int) {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        val cm = ctx.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val caps = ShadowNetworkCapabilities.newInstance()
        shadowOf(caps).addTransportType(transport)
        shadowOf(cm).setNetworkCapabilities(cm.activeNetwork, caps)
    }
}
