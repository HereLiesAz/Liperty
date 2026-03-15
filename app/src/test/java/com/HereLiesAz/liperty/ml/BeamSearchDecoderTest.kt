package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class BeamSearchDecoderTest {

    @Test
    fun testDecodeSimple() {
        val decoder = BeamSearchDecoder(beamWidth = 3)
        val vocabSize = 40 // _, 39 phonemes

        // Sequence: AA, _, AE
        val probs = Array(3) { FloatArray(vocabSize) }

        // Step 1: High prob for AA (1)
        probs[0][1] = 0.9f
        probs[0][0] = 0.1f

        // Step 2: High prob for Blank (0)
        probs[1][0] = 0.9f
        probs[1][1] = 0.1f

        // Step 3: High prob for AE (2)
        probs[2][2] = 0.9f
        probs[2][0] = 0.1f

        val result = decoder.decode(probs)
        assertEquals("AAAE", result)
    }

    @Test
    fun testDecodeRepeatedWithBlank() {
        val decoder = BeamSearchDecoder(beamWidth = 3)
        val vocabSize = 40

        // Sequence: AA, _, AA -> "AAAA"
        val probs = Array(3) { FloatArray(vocabSize) }

        probs[0][1] = 0.9f
        probs[1][0] = 0.9f
        probs[2][1] = 0.9f

        val result = decoder.decode(probs)
        assertEquals("AAAA", result)
    }

    @Test
    fun testDecodeRepeatedWithoutBlank() {
        val decoder = BeamSearchDecoder(beamWidth = 3)
        val vocabSize = 40

        // Sequence: AA, AA -> "AA" (Collapsed)
        val probs = Array(2) { FloatArray(vocabSize) }

        probs[0][1] = 0.9f
        probs[1][1] = 0.9f

        val result = decoder.decode(probs)
        assertEquals("AA", result)
    }
}
