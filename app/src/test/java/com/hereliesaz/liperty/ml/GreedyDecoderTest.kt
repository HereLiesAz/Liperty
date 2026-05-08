package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class GreedyDecoderTest {

    @Test
    fun testDecodeSimple() {
        val decoder = GreedyDecoder()
        val vocabSize = 40 // _, 39 phonemes

        val probs = Array(3) { FloatArray(vocabSize) }
        probs[0][1] = 1.0f // AA
        probs[1][0] = 1.0f // _
        probs[2][2] = 1.0f // AE

        val result = decoder.decode(probs)
        assertEquals("AA AE", result)
    }

    @Test
    fun testDecodeRepeated() {
        val decoder = GreedyDecoder()
        val vocabSize = 40

        val probs = Array(3) { FloatArray(vocabSize) }
        probs[0][1] = 1.0f // AA
        probs[1][1] = 1.0f // AA
        probs[2][2] = 1.0f // AE

        val result = decoder.decode(probs)
        assertEquals("AA AE", result)
    }

    @Test
    fun testDecodeRepeatedWithBlank() {
        val decoder = GreedyDecoder()
        val vocabSize = 40

        val probs = Array(3) { FloatArray(vocabSize) }
        probs[0][1] = 1.0f // AA
        probs[1][0] = 1.0f // _
        probs[2][1] = 1.0f // AA

        val result = decoder.decode(probs)
        assertEquals("AA AA", result)
    }
}
