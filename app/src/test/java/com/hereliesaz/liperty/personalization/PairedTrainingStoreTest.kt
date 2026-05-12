package com.hereliesaz.liperty.personalization

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import java.io.File
import java.nio.file.Files

/**
 * Tests for [PairedTrainingStore] — on-disk persistence for the paired
 * (audio, video, transcript) recordings used to personalize Liperty's
 * visual ASR per-user.
 *
 * The store wraps a single directory. All persistence is binary +
 * minimal-JSON; we don't depend on Android's ContentProvider so the
 * tests can run as pure JVM with a tempdir.
 *
 * Key behaviors:
 *   - save() round-trips: load() returns the same content.
 *   - listAll() enumerates known records by id.
 *   - delete(id) removes physically (the binary files, not just the index).
 *   - deleteAll() wipes the entire directory contents — biometric data
 *     must be fully removable on user request.
 *   - totalBytes() exposes the on-disk footprint for the UI.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class PairedTrainingStoreTest {

    private lateinit var tempDir: File

    @Before
    fun setUp() {
        tempDir = Files.createTempDirectory("liperty-paired-test").toFile()
    }

    @After
    fun tearDown() {
        tempDir.deleteRecursively()
    }

    private fun makeRecord(id: String, transcript: String? = null) =
        PairedTrainingRecord(
            id = id,
            audioPcm = FloatArray(16_000) { (it % 100) / 100f },          // 1 sec
            videoFrames = List(25) { FloatArray(88 * 88) { (it % 10) / 10f } }, // 1 sec @ 25 fps
            createdAtMs = 1_700_000_000_000L + id.hashCode(),
            source = "voice_import_${id}.mp4",
            transcript = transcript,
        )

    @Test
    fun saveAndLoadRoundTripsContent() {
        val store = PairedTrainingStore(tempDir)
        val rec = makeRecord("abc123", "the quick brown fox")
        store.save(rec)
        val loaded = store.load("abc123")
        assertNotNull(loaded)
        assertEquals(rec, loaded)
    }

    @Test
    fun loadOfMissingIdReturnsNull() {
        val store = PairedTrainingStore(tempDir)
        assertEquals(null, store.load("does-not-exist"))
    }

    @Test
    fun listAllEnumeratesSavedIds() {
        val store = PairedTrainingStore(tempDir)
        store.save(makeRecord("a"))
        store.save(makeRecord("b"))
        store.save(makeRecord("c"))
        assertEquals(setOf("a", "b", "c"), store.listAll().toSet())
    }

    @Test
    fun deleteRemovesPhysicalFiles() {
        val store = PairedTrainingStore(tempDir)
        store.save(makeRecord("temp"))
        assertTrue(tempDir.listFiles().orEmpty().any { it.name.startsWith("temp") })
        store.delete("temp")
        assertFalse(tempDir.listFiles().orEmpty().any { it.name.startsWith("temp") })
        assertEquals(null, store.load("temp"))
    }

    @Test
    fun deleteAllWipesEverything() {
        val store = PairedTrainingStore(tempDir)
        store.save(makeRecord("x"))
        store.save(makeRecord("y"))
        assertEquals(2, store.listAll().size)
        store.deleteAll()
        assertEquals(0, store.listAll().size)
        // Defensive: the directory itself still exists (Liperty needs it to
        // re-populate on next training session), but it's empty.
        assertTrue(tempDir.exists())
        assertEquals(0, tempDir.listFiles()?.size ?: 0)
    }

    @Test
    fun totalBytesReportsOnDiskFootprint() {
        val store = PairedTrainingStore(tempDir)
        val empty = store.totalBytes()
        assertEquals(0L, empty)
        store.save(makeRecord("rec1"))
        val one = store.totalBytes()
        assertTrue("one record > 0 bytes", one > 0)
        store.save(makeRecord("rec2"))
        val two = store.totalBytes()
        assertTrue("two records > one record", two > one)
    }

    @Test
    fun savePreservesOptionalTranscriptField() {
        val store = PairedTrainingStore(tempDir)
        val withT = makeRecord("with", "hello world")
        val withoutT = makeRecord("without", null)
        store.save(withT)
        store.save(withoutT)
        assertEquals("hello world", store.load("with")!!.transcript)
        assertEquals(null, store.load("without")!!.transcript)
    }

    @Test
    fun saveOverwritesExistingRecordOfSameId() {
        val store = PairedTrainingStore(tempDir)
        store.save(makeRecord("x", "first"))
        store.save(makeRecord("x", "second"))
        assertEquals("second", store.load("x")!!.transcript)
    }
}
