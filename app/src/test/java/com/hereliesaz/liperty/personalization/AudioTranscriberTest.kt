package com.hereliesaz.liperty.personalization

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Tests for [AudioTranscriber] — the interface that supplies transcript
 * labels to [PairedTrainingRecord]. Real implementations wrap an
 * on-device ASR (Vosk and Whisper.cpp are the leading candidates — see
 * [NoopAudioTranscriber]'s KDoc for the integration plan); this file
 * only covers the interface contract and the no-op fallback used when
 * no ASR backend is available.
 *
 * The transcript field is intentionally OPTIONAL on the record. Step 2
 * personalization layers that need text (personal n-gram LM, encoder
 * LoRA's CTC supervision) filter samples by `transcript != null`;
 * unsupervised layers (viseme confusion statistics, audio-only voice
 * cloning) ignore the field entirely.
 */
class AudioTranscriberTest {

    @Test
    fun noopTranscriberReturnsNullForAnyAudio() {
        val t = NoopAudioTranscriber()
        val result = t.transcribe(FloatArray(16_000) { it.toFloat() / 16_000f }, sampleRate = 16_000)
        assertNull(result)
    }

    @Test
    fun noopTranscriberReturnsNullForEmptyAudio() {
        val t = NoopAudioTranscriber()
        assertNull(t.transcribe(FloatArray(0), sampleRate = 16_000))
    }

    @Test
    fun noopTranscriberCloseIsIdempotent() {
        val t = NoopAudioTranscriber()
        t.close()
        t.close()  // second close must not throw
    }

    @Test
    fun transcriptionResultExposesTextAndConfidence() {
        // Sanity check on the data class itself.
        val r = TranscriptionResult(text = "hello world", confidence = 0.87f)
        assertEquals("hello world", r.text)
        assertEquals(0.87f, r.confidence, 0f)
    }
}
