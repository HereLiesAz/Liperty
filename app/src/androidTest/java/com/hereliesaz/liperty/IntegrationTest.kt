package com.hereliesaz.liperty

import androidx.lifecycle.Lifecycle
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The UI is Jetpack Compose with programmatically created views, so there are no
 * XML view ids (R.id.viewFinder etc.) to assert on. This smoke test verifies the
 * activity launches and reaches a started state on device; the consent dialog
 * may be showing on a fresh install, which is expected.
 */
@RunWith(AndroidJUnit4::class)
class IntegrationTest {

    @get:Rule
    val activityRule = ActivityScenarioRule(MainActivity::class.java)

    @Test
    fun testAppLaunch() {
        assertTrue(activityRule.scenario.state.isAtLeast(Lifecycle.State.STARTED))
    }
}
