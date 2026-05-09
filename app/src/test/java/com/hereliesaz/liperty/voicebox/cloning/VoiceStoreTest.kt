package com.hereliesaz.liperty.voicebox.cloning

import com.hereliesaz.liperty.voicebox.VoiceState
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

@RunWith(RobolectricTestRunner::class)
class VoiceStoreTest {

    private val store = VoiceStore(RuntimeEnvironment.getApplication())

    // ── Helpers ──────────────────────────────────────────────────────────

    private fun makeEmbedding(seed: Float = 1f, size: Int = 256): FloatArray =
        FloatArray(size) { i -> seed + i * 0.01f }

    private fun makeSample(
        id: String = "sample-1",
        fileName: String = "recording.wav",
        durationMs: Long = 3000,
        embedding: FloatArray = makeEmbedding(),
        snr: Float = 18f,
        addedAtMs: Long = System.currentTimeMillis()
    ) = VoiceSample(
        id = id,
        sourceFileName = fileName,
        durationMs = durationMs,
        embedding = embedding,
        snrEstimate = snr,
        addedAtMs = addedAtMs
    )

    private fun makeProfile(
        name: String = "Alice",
        createdAtMs: Long = 1_700_000_000_000L,
        lastUpdatedMs: Long = 1_700_000_060_000L,
        samples: List<VoiceSample> = listOf(
            makeSample(id = "s1", fileName = "clip1.wav", durationMs = 2500, embedding = makeEmbedding(1f), snr = 20f, addedAtMs = 1_700_000_010_000L),
            makeSample(id = "s2", fileName = "clip2.wav", durationMs = 4000, embedding = makeEmbedding(2f), snr = 15f, addedAtMs = 1_700_000_030_000L),
            makeSample(id = "s3", fileName = "clip3.wav", durationMs = 1800, embedding = makeEmbedding(3f), snr = 22f, addedAtMs = 1_700_000_050_000L)
        ),
        averagedEmbedding: FloatArray = makeEmbedding(5f),
        qualityScore: Float = 0.72f
    ) = VoiceProfile(
        name = name,
        createdAtMs = createdAtMs,
        lastUpdatedMs = lastUpdatedMs,
        samples = samples,
        averagedEmbedding = averagedEmbedding,
        qualityScore = qualityScore
    )

    // ── Tests ────────────────────────────────────────────────────────────

    @Test
    fun `saveProfile and loadProfile roundtrip`() {
        val original = makeProfile(name = "RoundTrip")
        store.saveProfile(original)

        val loaded = store.loadProfile("RoundTrip")
        assertNotNull(loaded)
        loaded!!

        assertEquals(original.name, loaded.name)
        assertEquals(original.createdAtMs, loaded.createdAtMs)
        assertEquals(original.lastUpdatedMs, loaded.lastUpdatedMs)
        assertEquals(original.qualityScore, loaded.qualityScore, 0.001f)
        assertArrayEquals(original.averagedEmbedding, loaded.averagedEmbedding, 0.001f)

        // Verify each sample
        assertEquals(original.samples.size, loaded.samples.size)
        for (i in original.samples.indices) {
            val os = original.samples[i]
            val ls = loaded.samples[i]
            assertEquals(os.id, ls.id)
            assertEquals(os.sourceFileName, ls.sourceFileName)
            assertEquals(os.durationMs, ls.durationMs)
            assertEquals(os.snrEstimate, ls.snrEstimate, 0.001f)
            assertEquals(os.addedAtMs, ls.addedAtMs)
            assertArrayEquals(os.embedding, ls.embedding, 0.001f)
        }
    }

    @Test
    fun `loadAllProfiles returns all saved profiles`() {
        val profileA = makeProfile(name = "ProfileA", qualityScore = 0.6f)
        val profileB = makeProfile(name = "ProfileB", qualityScore = 0.9f)
        store.saveProfile(profileA)
        store.saveProfile(profileB)

        val all = store.loadAllProfiles()
        val names = all.map { it.name }.toSet()

        assertEquals(2, all.size)
        assertTrue("ProfileA" in names)
        assertTrue("ProfileB" in names)
    }

    @Test
    fun `loadAllVoices merges legacy vstate with vprofile`() {
        // Save a legacy voice only (no profile)
        val legacyEmbedding = makeEmbedding(seed = 10f)
        store.saveVoice(VoiceState("LegacyVoice", legacyEmbedding))

        // Save a profile (which also creates a legacy .vstate, but we test distinct names)
        val profile = makeProfile(name = "ProfileVoice")
        store.saveProfile(profile)

        val allVoices = store.loadAllVoices()
        val names = allVoices.map { it.name }.toSet()

        assertTrue("Should contain the legacy voice", "LegacyVoice" in names)
        assertTrue("Should contain the profile voice", "ProfileVoice" in names)
        assertTrue("Should have at least 2 voices", allVoices.size >= 2)
    }

    @Test
    fun `deleteProfile removes directory`() {
        val profile = makeProfile(name = "ToDelete")
        store.saveProfile(profile)

        // Confirm it exists first
        assertNotNull(store.loadProfile("ToDelete"))

        store.deleteProfile("ToDelete")

        assertNull(store.loadProfile("ToDelete"))
    }

    @Test
    fun `deleteVoice removes both legacy and profile`() {
        val profile = makeProfile(name = "FullDelete")
        store.saveProfile(profile) // saveProfile also writes a legacy .vstate

        // Confirm both exist
        assertNotNull(store.loadProfile("FullDelete"))
        assertTrue(store.loadAllVoices().any { it.name == "FullDelete" })

        store.deleteVoice("FullDelete")

        // Profile should be gone
        assertNull(store.loadProfile("FullDelete"))
        // Legacy voice should also be gone
        assertTrue(store.loadAllVoices().none { it.name == "FullDelete" })
    }

    @Test
    fun `loadProfile returns null for nonexistent`() {
        assertNull(store.loadProfile("NeverSaved"))
    }
}
