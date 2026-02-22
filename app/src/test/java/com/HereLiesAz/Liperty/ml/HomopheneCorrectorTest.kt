package com.HereLiesAz.Liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HomopheneCorrectorTest {

    private val corrector = HomopheneCorrector()

    @Test
    fun testGetAlternativesKnownWord() {
        val alternatives = corrector.getAlternatives("mat")
        // "bat", "pat" should be in the list
        assertTrue(alternatives.contains("bat"))
        assertTrue(alternatives.contains("pat"))
    }

    @Test
    fun testGetAlternativesCaseInsensitive() {
        val alternatives = corrector.getAlternatives("Mat")
        assertTrue(alternatives.contains("bat"))
    }

    @Test
    fun testGetAlternativesUnknownWord() {
        val alternatives = corrector.getAlternatives("unknown")
        assertTrue(alternatives.isEmpty())
    }
}
