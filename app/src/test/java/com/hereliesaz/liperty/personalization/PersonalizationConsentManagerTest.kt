package com.hereliesaz.liperty.personalization

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/**
 * Tests for [PersonalizationConsentManager] — the persistence layer for
 * the *separate* opt-in to on-device personalization data collection.
 *
 * Distinct from the app-launch consent gate ([com.hereliesaz.liperty.MainActivity.checkConsentAndStart]):
 * a user may agree to use Liperty without also agreeing to have video
 * of themselves stored on-device for training. Personalization consent
 * is a second, narrower decision and must be revocable independently.
 *
 * Revocation invariant: revoking consent triggers physical deletion of
 * all stored [PairedTrainingRecord]s. The consent flag and the data
 * lifecycle are coupled.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class PersonalizationConsentManagerTest {

    @Before
    fun clearPrefs() {
        // Each test gets a fresh consent state.
        val ctx = ApplicationProvider.getApplicationContext<android.content.Context>()
        ctx.getSharedPreferences(PersonalizationConsentManager.PREFS_NAME, 0)
            .edit().clear().commit()
    }

    @Test
    fun defaultsToNotGranted() {
        val ctx = ApplicationProvider.getApplicationContext<android.content.Context>()
        val cm = PersonalizationConsentManager(ctx)
        assertFalse("no consent before user opts in", cm.hasConsent())
    }

    @Test
    fun grantedConsentPersists() {
        val ctx = ApplicationProvider.getApplicationContext<android.content.Context>()
        PersonalizationConsentManager(ctx).grantConsent()
        // Fresh instance — simulates an app restart.
        val cm2 = PersonalizationConsentManager(ctx)
        assertTrue(cm2.hasConsent())
    }

    @Test
    fun revokeClearsConsent() {
        val ctx = ApplicationProvider.getApplicationContext<android.content.Context>()
        val cm = PersonalizationConsentManager(ctx)
        cm.grantConsent()
        cm.revokeConsent()
        assertFalse(cm.hasConsent())
    }

    @Test
    fun revokeAlsoTriggersDataDeletion() {
        val ctx = ApplicationProvider.getApplicationContext<android.content.Context>()
        val cm = PersonalizationConsentManager(ctx)
        var deletedCount = 0
        cm.onRevoke = { deletedCount++ }
        cm.grantConsent()
        cm.revokeConsent()
        assertEquals("revoke must invoke the data-deletion callback", 1, deletedCount)
    }

    @Test
    fun grantedAtTimestampIsRecorded() {
        val ctx = ApplicationProvider.getApplicationContext<android.content.Context>()
        val cm = PersonalizationConsentManager(ctx)
        val before = System.currentTimeMillis()
        cm.grantConsent()
        val ts = cm.grantedAtMs()
        assertTrue("timestamp recorded", ts != null)
        assertTrue("timestamp is recent", ts!! >= before)
    }

    @Test
    fun grantedAtTimestampIsNullBeforeConsent() {
        val ctx = ApplicationProvider.getApplicationContext<android.content.Context>()
        val cm = PersonalizationConsentManager(ctx)
        assertEquals(null, cm.grantedAtMs())
    }
}
