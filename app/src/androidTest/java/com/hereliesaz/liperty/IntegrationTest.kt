package com.hereliesaz.liperty

import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.withId
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class IntegrationTest {

    @get:Rule
    val activityRule = ActivityScenarioRule(MainActivity::class.java)

    @Test
    fun testAppLaunch() {
        // Verify Camera Preview
        onView(withId(R.id.viewFinder)).check(matches(isDisplayed()))

        // Verify Overlay
        onView(withId(R.id.overlay)).check(matches(isDisplayed()))

        // Verify Transcription Text
        onView(withId(R.id.text_transcription)).check(matches(isDisplayed()))

        // Verify Settings Button
        onView(withId(R.id.btn_settings)).check(matches(isDisplayed()))

        // Verify Recording Indicator (might take a moment to appear after camera start, but usually fast enough in tests)
        onView(withId(R.id.indicator_recording)).check(matches(isDisplayed()))
    }
}
