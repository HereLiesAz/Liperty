package com.HereLiesAz.Liperty.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HomopheneCorrectorTest {

    private val corrector = HomopheneCorrector()

    @Test
    fun testGetAlternatives() {
        val alternatives = corrector.getAlternatives("mat")
        assertTrue(alternatives.contains("bat"))
        assertTrue(alternatives.contains("pat"))
        assertEquals(2, alternatives.size)
    }

    @Test
    fun testGetAlternativesCaseInsensitive() {
        val alternatives = corrector.getAlternatives("Mat")
        assertTrue(alternatives.contains("bat"))
    }

    @Test
    fun testNoAlternatives() {
        val alternatives = corrector.getAlternatives("unique")
        assertTrue(alternatives.isEmpty())
    }

    @Test
    fun testAreHomophenes() {
        assertTrue(corrector.areHomophenes("mat", "bat"))
        assertTrue(!corrector.areHomophenes("mat", "cat")) // Assuming 'cat' (k) is not homophene with 'mat' (m) in this simple dictionary
    }
}
