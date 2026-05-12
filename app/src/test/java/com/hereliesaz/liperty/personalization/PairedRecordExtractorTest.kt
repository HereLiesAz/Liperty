package com.hereliesaz.liperty.personalization

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Tests for [PairedRecordExtractor] — the orchestrator that combines
 * audio segments (from `AudioPreprocessor`), synchronized video frames
 * (from `VideoFrameExtractor`), and optional transcripts (from
 * `AudioTranscriber`) into [PairedTrainingRecord]s for [PairedTrainingStore].
 *
 * Tests use lambda-injected providers so the orchestrator can be
 * exercised without real Android media APIs. The pure logic verified:
 *   - emits one record per audio segment
 *   - aligns video frames by audio segment's `startTimeMs/durationMs`
 *   - transcript field is null when transcriber returns null
 *   - skips segments whose video frame count is 0 (no video track / extraction failed)
 *   - threading through the record's id + source metadata is intact
 */
class PairedRecordExtractorTest {

    private fun audioSegment(start: Long, duration: Long) =
        PairedRecordExtractor.AudioInput(
            pcm = FloatArray((duration * 16).toInt()) { 0.5f },   // 16 samples/ms
            startTimeMs = start,
            durationMs = duration,
        )

    @Test
    fun emitsOneRecordPerAudioSegment() {
        val extractor = PairedRecordExtractor(
            extractVideoFrames = { _, _, _ -> List(4) { FloatArray(88 * 88) { 0.1f } } },
            transcriber = NoopAudioTranscriber(),
        )
        val segments = listOf(
            audioSegment(start = 0L, duration = 800L),
            audioSegment(start = 1000L, duration = 600L),
            audioSegment(start = 2000L, duration = 1200L),
        )
        val records = extractor.extract(segments, sourceUri = "test.mp4")
        assertEquals(3, records.size)
    }

    @Test
    fun nullTranscriberProducesNullTranscriptField() {
        val extractor = PairedRecordExtractor(
            extractVideoFrames = { _, _, _ -> List(4) { FloatArray(88 * 88) } },
            transcriber = NoopAudioTranscriber(),
        )
        val rec = extractor.extract(listOf(audioSegment(0L, 1000L)), "test.mp4").first()
        assertNull(rec.transcript)
        assertNull(rec.transcriptConfidence)
    }

    @Test
    fun activeTranscriberPopulatesTranscriptAndConfidence() {
        val extractor = PairedRecordExtractor(
            extractVideoFrames = { _, _, _ -> List(4) { FloatArray(88 * 88) } },
            transcriber = object : AudioTranscriber {
                override fun transcribe(audioPcm: FloatArray, sampleRate: Int) =
                    TranscriptionResult("hello world", 0.85f)
                override fun close() {}
            },
        )
        val rec = extractor.extract(listOf(audioSegment(0L, 1000L)), "test.mp4").first()
        assertEquals("hello world", rec.transcript)
        assertEquals(0.85f, rec.transcriptConfidence!!, 0f)
    }

    @Test
    fun videoExtractorCalledWithSegmentStartAndDuration() {
        val received = ArrayList<Pair<Long, Long>>()
        val extractor = PairedRecordExtractor(
            extractVideoFrames = { _, start, dur ->
                received.add(start to dur)
                List(2) { FloatArray(88 * 88) }
            },
            transcriber = NoopAudioTranscriber(),
        )
        extractor.extract(
            listOf(
                audioSegment(start = 100L, duration = 500L),
                audioSegment(start = 2000L, duration = 800L),
            ),
            "test.mp4",
        )
        assertEquals(listOf(100L to 500L, 2000L to 800L), received)
    }

    @Test
    fun segmentsWithZeroExtractedFramesAreSkipped() {
        // No video track in the source, or extraction failed — drop the
        // segment rather than store an unusable record with empty video.
        var callCount = 0
        val extractor = PairedRecordExtractor(
            extractVideoFrames = { _, _, _ ->
                callCount++
                emptyList()
            },
            transcriber = NoopAudioTranscriber(),
        )
        val records = extractor.extract(
            listOf(audioSegment(0L, 1000L), audioSegment(2000L, 1000L)),
            "test.mp4",
        )
        assertEquals(0, records.size)
        assertEquals(2, callCount)
    }

    @Test
    fun recordsCarrySourceUriInMetadata() {
        val extractor = PairedRecordExtractor(
            extractVideoFrames = { _, _, _ -> List(2) { FloatArray(88 * 88) } },
            transcriber = NoopAudioTranscriber(),
        )
        val rec = extractor.extract(listOf(audioSegment(0L, 500L)), "alice_recording.mp4").first()
        assertEquals("alice_recording.mp4", rec.source)
    }

    @Test
    fun recordIdsAreUnique() {
        val extractor = PairedRecordExtractor(
            extractVideoFrames = { _, _, _ -> List(1) { FloatArray(88 * 88) } },
            transcriber = NoopAudioTranscriber(),
        )
        val records = extractor.extract(
            List(5) { audioSegment(it * 1000L, 500L) },
            "test.mp4",
        )
        val ids = records.map { it.id }.toSet()
        assertEquals(5, ids.size)
    }
}

