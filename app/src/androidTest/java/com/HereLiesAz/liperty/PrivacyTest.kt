package com.hereliesaz.liperty

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class PrivacyTest {

    @Test
    fun testNoTranscriptionHistoryPersisted() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val prefs = context.getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)

        // Ensure keys related to user data (like "history", "transcript") are NOT present.
        // Only settings should be there.
        assertFalse(prefs.contains("history"))
        assertFalse(prefs.contains("transcript"))
        assertFalse(prefs.contains("last_sentence"))

        // Verify valid keys are present (optional, but good for sanity)
        // assertTrue(prefs.contains("font_size")) // Might not be set yet if fresh install
    }

    @Test
    fun testNoVideoFilesStored() {
        val context = ApplicationProvider.getApplicationContext<Context>()

        // Check external files dir (where videos usually go)
        val externalDir = context.getExternalFilesDir(null)
        if (externalDir != null && externalDir.exists()) {
            val files = externalDir.walkTopDown().filter { it.isFile && (it.extension == "mp4" || it.extension == "mkv") }.toList()
            assertTrue("Found persisting video files: $files", files.isEmpty())
        }

        // Check cache dir
        val cacheDir = context.cacheDir
        if (cacheDir != null && cacheDir.exists()) {
             val files = cacheDir.walkTopDown().filter { it.isFile && (it.extension == "mp4" || it.extension == "mkv") }.toList()
             assertTrue("Found video files in cache: $files", files.isEmpty())
        }
    }
}
