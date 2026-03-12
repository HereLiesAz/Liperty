package com.hereliesaz.liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class LanguageModelTest {

    @Test
    fun testGetScore() {
        val lm = LanguageModel()

        // H -> E should be 0.5
        val score = lm.getScore("H", "E")
        assertEquals(0.5, score, 0.001)
    }

    @Test
    fun testGetScoreUnknown() {
        val lm = LanguageModel()

        // H -> Z should be 0.0
        val score = lm.getScore("H", "Z")
        assertEquals(0.0, score, 0.001)
    }
}
