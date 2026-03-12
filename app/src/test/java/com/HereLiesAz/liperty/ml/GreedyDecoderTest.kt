package com.HereLiesAz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class GreedyDecoderTest {

    @Test
    fun testDecodeSimple() {
        val decoder = GreedyDecoder()
        val vocabSize = 28 // _, A..Z, space

        val probs = Array(3) { FloatArray(vocabSize) }
        probs[0][1] = 1.0f // A
        probs[1][0] = 1.0f // _
        probs[2][2] = 1.0f // B

        val result = decoder.decode(probs)
        assertEquals("AB", result)
    }

    @Test
    fun testDecodeRepeated() {
        val decoder = GreedyDecoder()
        val vocabSize = 28

        val probs = Array(3) { FloatArray(vocabSize) }
        probs[0][1] = 1.0f // A
        probs[1][1] = 1.0f // A
        probs[2][2] = 1.0f // B

        val result = decoder.decode(probs)
        assertEquals("AB", result)
    }

    @Test
    fun testDecodeRepeatedWithBlank() {
        val decoder = GreedyDecoder()
        val vocabSize = 28

        val probs = Array(3) { FloatArray(vocabSize) }
        probs[0][1] = 1.0f // A
        probs[1][0] = 1.0f // _
        probs[2][1] = 1.0f // A

        val result = decoder.decode(probs)
        assertEquals("AA", result)
    }
}
