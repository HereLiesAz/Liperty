package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class LanguageModelTest {

    @Test
    fun testGetScore() {
        val lm = LanguageModel()

        // "HH" -> "IH" according to bigramMap should be 0.13
        val score = lm.getScore("HH", "IH")
        assertEquals(0.13, score, 0.001)
    }

    @Test
    fun testGetScoreUnknown() {
        val lm = LanguageModel()

        // "HH" -> "Z" is not in the map, so it should be UNIFORM_PROB (0.003)
        val score = lm.getScore("HH", "Z")
        assertEquals(0.003, score, 0.001)
    }
}
