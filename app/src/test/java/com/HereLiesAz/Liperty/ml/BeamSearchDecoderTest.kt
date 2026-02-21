package com.HereLiesAz.Liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class BeamSearchDecoderTest {

    @Test
    fun testDecodeSimple() {
        val decoder = BeamSearchDecoder(beamWidth = 3)
        val vocabSize = 28 // _, A..Z, space

        // Sequence: A, _, B
        val probs = Array(3) { FloatArray(vocabSize) }

        // Step 1: High prob for A (1)
        probs[0][1] = 0.9f
        probs[0][0] = 0.1f

        // Step 2: High prob for Blank (0)
        probs[1][0] = 0.9f
        probs[1][1] = 0.1f

        // Step 3: High prob for B (2)
        probs[2][2] = 0.9f
        probs[2][0] = 0.1f

        val result = decoder.decode(probs)
        assertEquals("AB", result)
    }

    @Test
    fun testDecodeRepeatedWithBlank() {
        val decoder = BeamSearchDecoder(beamWidth = 3)
        val vocabSize = 28

        // Sequence: A, _, A -> "AA"
        val probs = Array(3) { FloatArray(vocabSize) }

        probs[0][1] = 0.9f
        probs[1][0] = 0.9f
        probs[2][1] = 0.9f

        val result = decoder.decode(probs)
        assertEquals("AA", result)
    }

    @Test
    fun testDecodeRepeatedWithoutBlank() {
        val decoder = BeamSearchDecoder(beamWidth = 3)
        val vocabSize = 28

        // Sequence: A, A -> "A" (Collapsed)
        val probs = Array(2) { FloatArray(vocabSize) }

        probs[0][1] = 0.9f
        probs[1][1] = 0.9f

        val result = decoder.decode(probs)
        assertEquals("A", result)
    }
}
