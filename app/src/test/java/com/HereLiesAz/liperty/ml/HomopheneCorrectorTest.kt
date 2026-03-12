package com.HereLiesAz.liperty.ml

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class HomopheneCorrectorTest {

    private lateinit var corrector: HomopheneCorrector
    private lateinit var context: Context

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        corrector = HomopheneCorrector(context)
    }

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
        val alternatives = corrector.getAlternatives("unknownwordXYZ")
        assertTrue(alternatives.isEmpty())
    }
}
