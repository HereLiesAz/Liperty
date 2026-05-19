package com.hereliesaz.liperty.ml

import com.hereliesaz.liperty.personalization.PairedTrainingRecord
import com.hereliesaz.liperty.personalization.PairedTrainingStore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.nio.file.Files

class TrainingDataPreparerTest {

    /** Fake encoder that returns deterministic features. */
    private class FakeEncoder : EncoderSession {
        override fun initialize() = true

        override fun runEncode(
            input: FloatArray,
            inputShape: LongArray,
        ): Pair<FloatArray, LongArray> {
            val T = inputShape[2].toInt()
            // Return simple features: T frames x 768 dims, filled with 0.5
            val features = FloatArray(T * 768) { 0.5f }
            return features to longArrayOf(1L, T.toLong(), 768L)
        }

        override fun close() {}
    }

    private lateinit var preparer: TrainingDataPreparer
    private lateinit var tempDir: File
    private lateinit var store: PairedTrainingStore

    @Before
    fun setUp() {
        preparer = TrainingDataPreparer(FakeEncoder())
        tempDir = Files.createTempDirectory("training_preparer_test").toFile()
        store = PairedTrainingStore(tempDir)
    }

    @Test
    fun prepareConvertsRecordToCorrectDimensions() {
        val numFrames = 10
        val record = makeRecord("test-1", numFrames)
        val sample = preparer.prepare(record)

        assertNotNull(sample)
        assertEquals(numFrames, sample!!.numFrames)
        assertEquals(numFrames * 88 * 88, sample.videoFrames.size)
        assertEquals(numFrames, sample.numTargetFrames)
        assertEquals(numFrames * 768, sample.targetFeatures.size)
    }

    @Test
    fun prepareReturnsNullForEmptyRecord() {
        val record = PairedTrainingRecord(
            id = "empty",
            audioPcm = FloatArray(0),
            videoFrames = emptyList(),
            createdAtMs = 0L,
            source = "test",
        )
        assertNull(preparer.prepare(record))
    }

    @Test
    fun prepareTruncatesToMaxFrames() {
        val smallPreparer = TrainingDataPreparer(FakeEncoder(), maxFrames = 10)
        val record = makeRecord("long", numFrames = 50)
        val sample = smallPreparer.prepare(record)

        assertNotNull(sample)
        assertEquals(10, sample!!.numFrames)
        assertEquals(10 * 88 * 88, sample.videoFrames.size)
    }

    @Test
    fun prepareTakesLastFramesWhenTruncating() {
        val smallPreparer = TrainingDataPreparer(FakeEncoder(), maxFrames = 2)
        // Create frames with distinguishable content
        val frames = (0 until 5).map { i ->
            FloatArray(88 * 88) { i.toFloat() }
        }
        val record = PairedTrainingRecord(
            id = "ordered",
            audioPcm = FloatArray(0),
            videoFrames = frames,
            createdAtMs = 0L,
            source = "test",
        )
        val sample = smallPreparer.prepare(record)

        assertNotNull(sample)
        assertEquals(2, sample!!.numFrames)
        // Should take last 2 frames (indices 3, 4)
        assertEquals(3f, sample.videoFrames[0], 0.001f)
        assertEquals(4f, sample.videoFrames[88 * 88], 0.001f)
    }

    @Test
    fun prepareAllFromStoreFiltersEmptyRecords() {
        store.save(makeRecord("good", 10))
        store.save(PairedTrainingRecord(
            id = "empty",
            audioPcm = FloatArray(0),
            videoFrames = emptyList(),
            createdAtMs = 0L,
            source = "test",
        ))
        store.save(makeRecord("good2", 5))

        val samples = preparer.prepareAll(store)
        assertEquals(2, samples.size)
    }

    @Test
    fun targetFeaturesArePopulated() {
        val record = makeRecord("feat-test", 8)
        val sample = preparer.prepare(record)

        assertNotNull(sample)
        // FakeEncoder fills with 0.5f
        assertTrue(sample!!.targetFeatures.all { it == 0.5f })
    }

    private fun makeRecord(id: String, numFrames: Int): PairedTrainingRecord {
        return PairedTrainingRecord(
            id = id,
            audioPcm = FloatArray(16_000), // 1 second of audio
            videoFrames = List(numFrames) { FloatArray(88 * 88) { 0.1f } },
            createdAtMs = System.currentTimeMillis(),
            source = "test",
        )
    }
}
