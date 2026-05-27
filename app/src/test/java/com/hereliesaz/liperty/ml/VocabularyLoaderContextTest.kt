package com.hereliesaz.liperty.ml

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import java.io.File

/**
 * Tests the filesDir-aware loaders added to fix the post-download crash:
 * vocab files are DOWNLOADED into [Context.getFilesDir], so the old
 * assets-only [VocabularyLoader.loadFromAssets] couldn't see them. [load] and
 * [readTextPreferringFiles] must prefer a downloaded file and fall back to a
 * bundled asset.
 *
 * AndroidJUnit4 + sdk=34 per the project's Robolectric/targetSdk=37 workaround.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class VocabularyLoaderContextTest {

    private fun context(): Context = ApplicationProvider.getApplicationContext()

    @Test
    fun loadPrefersDownloadedFileInFilesDir() {
        val ctx = context()
        val name = "test_unigram_units.txt"
        File(ctx.filesDir, name).writeText("▁hi 1\n▁there 2\n")

        val vocab = VocabularyLoader.load(ctx, name, blank = "<blank>")

        // Blank prepended at index 0, then tokens in file order.
        assertEquals(listOf("<blank>", "▁hi", "▁there"), vocab)
    }

    @Test
    fun readTextPrefersFilesDirOverAsset() {
        val ctx = context()
        // homophones.json IS a bundled asset; a filesDir file of the same name
        // must win.
        val name = "homophones.json"
        File(ctx.filesDir, name).writeText("FROM_FILES_DIR")

        assertEquals("FROM_FILES_DIR", VocabularyLoader.readTextPreferringFiles(ctx, name))
    }

    @Test
    fun readTextFallsBackToAssetWhenNoFile() {
        val ctx = context()
        // No filesDir copy of this bundled asset -> read from assets.
        File(ctx.filesDir, "homophones.json").delete()

        val text = VocabularyLoader.readTextPreferringFiles(ctx, "homophones.json")
        assertTrue("asset content should be non-empty JSON", text.contains("{"))
    }
}
