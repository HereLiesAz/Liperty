package com.HereLiesAz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class DecoderTest {

    @Test
    fun testGreedyDecoder() {
        val decoder = GreedyDecoder()

        // Vocab: 0=_, 1=A..8=H..5=E..12=L..15=O
        // Sequence: H, H, _, E, L, L, _, L, O -> HELLO
        // Indices: 8, 8, 0, 5, 12, 12, 0, 12, 15
        val indices = listOf(8, 8, 0, 5, 12, 12, 0, 12, 15)

        val vocabSize = 28
        val probabilities = Array(indices.size) { i ->
            val arr = FloatArray(vocabSize)
            arr[indices[i]] = 1.0f // Certainty
            arr
        }

        val result = decoder.decode(probabilities)
        assertEquals("HELLO", result)
    }

    @Test
    fun testGreedyDecoderWithRepeatedCharacters() {
        // "LL" should be decoded as "L" if no blank in between
        val decoder = GreedyDecoder()

        // L, L -> L
        val indices = listOf(12, 12)
        val probabilities = Array(indices.size) { i ->
            val arr = FloatArray(28)
            arr[indices[i]] = 1.0f
            arr
        }

        val result = decoder.decode(probabilities)
        assertEquals("L", result)
    }

    @Test
    fun testGreedyDecoderWithBlankBetweenRepeated() {
        // "L", "_", "L" -> "LL"
        val decoder = GreedyDecoder()

        val indices = listOf(12, 0, 12)
        val probabilities = Array(indices.size) { i ->
            val arr = FloatArray(28)
            arr[indices[i]] = 1.0f
            arr
        }

        val result = decoder.decode(probabilities)
        assertEquals("LL", result)
    }
}
