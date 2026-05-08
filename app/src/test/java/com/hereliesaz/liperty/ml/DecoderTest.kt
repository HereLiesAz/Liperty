package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class DecoderTest {

    @Test
    fun testGreedyDecoder() {
        val decoder = GreedyDecoder()

        // Test with new 40 phoneme vocabulary
        // Indices: 1=AA, 2=AE, 3=AH
        // Sequence: AA, AA, _, AE, AH, AH, _, AH, AE
        // Collapses to: AA, _, AE, AH, _, AH, AE
        // Blanks removed: AAAEAHAHAE
        val indices = listOf(1, 1, 0, 2, 3, 3, 0, 3, 2)

        val vocabSize = 40
        val probabilities = Array(indices.size) { i ->
            val arr = FloatArray(vocabSize)
            arr[indices[i]] = 1.0f // Certainty
            arr
        }

        val result = decoder.decode(probabilities)
        assertEquals("AA AE AH AH AE", result)
    }

    @Test
    fun testGreedyDecoderWithRepeatedCharacters() {
        val decoder = GreedyDecoder()

        // AH, AH -> AH
        val indices = listOf(3, 3)
        val probabilities = Array(indices.size) { i ->
            val arr = FloatArray(40)
            arr[indices[i]] = 1.0f
            arr
        }

        val result = decoder.decode(probabilities)
        assertEquals("AH", result)
    }

    @Test
    fun testGreedyDecoderWithBlankBetweenRepeated() {
        val decoder = GreedyDecoder()

        // AH, _, AH -> AHAH
        val indices = listOf(3, 0, 3)
        val probabilities = Array(indices.size) { i ->
            val arr = FloatArray(40)
            arr[indices[i]] = 1.0f
            arr
        }

        val result = decoder.decode(probabilities)
        assertEquals("AH AH", result)
    }
}
