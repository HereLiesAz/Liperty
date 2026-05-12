package com.hereliesaz.liperty.personalization

import android.content.Context
import android.content.SharedPreferences

/**
 * Persistence + business logic for the *separate* on-device-personalization
 * opt-in. Distinct from the app-launch consent gate ([com.hereliesaz.liperty.MainActivity.checkConsentAndStart]).
 *
 * Why two consents:
 *   The app-launch gate covers Liperty's base biometric processing
 *   (face landmarks, lip motion samples) — all in RAM, never persisted.
 *   Personalization REQUIRES persisting biometric data on-device for
 *   training (see [PairedTrainingRecord]). That's a different decision
 *   the user must be able to make — and revoke — independently.
 *
 * Revocation semantics (mandatory):
 *   When the user revokes personalization consent, all stored
 *   [PairedTrainingRecord]s must be physically deleted. The
 *   `[onRevoke]` callback is invoked synchronously inside [revokeConsent]
 *   for callers (typically MainActivity) to trigger the
 *   [PairedTrainingStore.deleteAll]. We don't take a direct dependency
 *   on the store so this class stays trivially unit-testable.
 *
 * Storage: [android.content.SharedPreferences] under [PREFS_NAME]. Two
 * fields:
 *   - `consent_granted` (Boolean): the consent state.
 *   - `consent_granted_at_ms` (Long): wall-clock when granted; useful
 *     for displaying "you opted in on <date>" in the Settings UI.
 */
class PersonalizationConsentManager(private val context: Context) {

    private val prefs: SharedPreferences by lazy {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    /** Optional callback invoked synchronously on [revokeConsent]. The
     *  caller wires this to [PairedTrainingStore.deleteAll] to enforce
     *  the "revoke = data deletion" invariant. */
    var onRevoke: (() -> Unit)? = null

    fun hasConsent(): Boolean = prefs.getBoolean(KEY_GRANTED, false)

    fun grantedAtMs(): Long? =
        if (hasConsent()) prefs.getLong(KEY_GRANTED_AT_MS, 0L).takeIf { it > 0 } else null

    fun grantConsent() {
        prefs.edit()
            .putBoolean(KEY_GRANTED, true)
            .putLong(KEY_GRANTED_AT_MS, System.currentTimeMillis())
            .apply()
    }

    fun revokeConsent() {
        prefs.edit()
            .putBoolean(KEY_GRANTED, false)
            .remove(KEY_GRANTED_AT_MS)
            .apply()
        onRevoke?.invoke()
    }

    companion object {
        const val PREFS_NAME = "liperty_personalization_consent"
        private const val KEY_GRANTED = "consent_granted"
        private const val KEY_GRANTED_AT_MS = "consent_granted_at_ms"
    }
}
