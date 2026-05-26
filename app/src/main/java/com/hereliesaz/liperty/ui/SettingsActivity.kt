package com.hereliesaz.liperty.ui

import android.content.Context
import android.os.Bundle
import android.widget.Switch
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.hereliesaz.liperty.R
import com.hereliesaz.liperty.personalization.PairedTrainingStore
import com.hereliesaz.liperty.personalization.PersonalizationConsentManager
import com.hereliesaz.liperty.personalization.TrainingDataExporter
import java.io.File

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        val switchTelephoto = findViewById<Switch>(R.id.switch_telephoto)
        val switchDarkTheme = findViewById<Switch>(R.id.switch_dark_theme)

        val sharedPrefs = getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)

        // Load saved preferences
        val savedTelephoto = sharedPrefs.getBoolean("telephoto_preference", true)
        val savedDarkTheme = sharedPrefs.getBoolean("dark_theme", true)

        switchTelephoto.isChecked = savedTelephoto
        switchDarkTheme.isChecked = savedDarkTheme

        switchTelephoto.setOnCheckedChangeListener { _, isChecked ->
            sharedPrefs.edit().putBoolean("telephoto_preference", isChecked).apply()
        }

        switchDarkTheme.setOnCheckedChangeListener { _, isChecked ->
            sharedPrefs.edit().putBoolean("dark_theme", isChecked).apply()
            // Apply immediately so the surface flips without an app restart.
            AppCompatDelegate.setDefaultNightMode(
                if (isChecked) AppCompatDelegate.MODE_NIGHT_YES
                else AppCompatDelegate.MODE_NIGHT_NO
            )
        }

        // Privacy / consent + personalization data management (Compose).
        val personalizationConsent = PersonalizationConsentManager(this)
        val storeDir = File(filesDir, "viseme_calibration")
        val store = PairedTrainingStore(storeDir)
        // Enforce "revoke personalization == delete persisted biometric data".
        personalizationConsent.onRevoke = { store.deleteAll() }

        findViewById<ComposeView>(R.id.compose_settings).setContent {
            MaterialTheme(
                colorScheme = if (savedDarkTheme) darkColorScheme() else lightColorScheme(),
            ) {
                Column(modifier = Modifier.fillMaxWidth()) {
                    BiometricConsentSection(
                        isGranted = sharedPrefs.getBoolean("consent_granted", false),
                        onRevoke = {
                            // Revoke the app-launch biometric-processing consent.
                            sharedPrefs.edit().putBoolean("consent_granted", false).apply()
                            // Personalization is a superset of base biometric
                            // processing, so revoking the base consent also
                            // revokes personalization and deletes its stored
                            // biometric data (via onRevoke -> store.deleteAll()).
                            if (personalizationConsent.hasConsent()) {
                                personalizationConsent.revokeConsent()
                            } else {
                                store.deleteAll()
                            }
                            // Leave Settings; MainActivity.onResume re-engages the
                            // consent gate (camera stays off until re-granted).
                            finish()
                        },
                    )

                    Spacer(Modifier.height(8.dp))

                    // Mirror the consent/store state into Compose state so the
                    // panel updates after grant/revoke/delete without a reload.
                    var consentGranted by remember {
                        mutableStateOf(personalizationConsent.hasConsent())
                    }
                    var recordCount by remember { mutableIntStateOf(store.listAll().size) }
                    var totalBytes by remember { mutableStateOf(store.totalBytes()) }

                    PersonalizationSettingsPanel(
                        consentGranted = consentGranted,
                        consentGrantedAtMs = personalizationConsent.grantedAtMs(),
                        recordCount = recordCount,
                        totalBytes = totalBytes,
                        onGrantConsent = {
                            personalizationConsent.grantConsent()
                            consentGranted = true
                        },
                        onRevokeConsent = {
                            personalizationConsent.revokeConsent() // deletes data
                            consentGranted = false
                            recordCount = store.listAll().size
                            totalBytes = store.totalBytes()
                        },
                        onDeleteAll = {
                            store.deleteAll()
                            recordCount = store.listAll().size
                            totalBytes = store.totalBytes()
                        },
                        onExportForTraining = {
                            TrainingDataExporter.export(this@SettingsActivity, storeDir)
                        },
                    )
                }
            }
        }
    }
}

/**
 * The mandatory app-launch biometric-processing consent, surfaced here so the
 * user can revoke it (BIPA/GDPR right to withdraw). Revoking stops all camera
 * processing; the gate re-engages on the next launch / resume.
 */
@Composable
private fun BiometricConsentSection(
    isGranted: Boolean,
    onRevoke: () -> Unit,
) {
    var showConfirm by remember { mutableStateOf(false) }

    Card(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                "Biometric processing",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                if (isGranted) {
                    "On — Liperty reads your lip movements (biometric data) in memory " +
                        "to transcribe speech. Nothing is stored or leaves this device."
                } else {
                    "Off — consent has not been granted. Lip-reading is disabled."
                },
                style = MaterialTheme.typography.bodySmall,
            )
            if (isGranted) {
                Spacer(Modifier.height(16.dp))
                OutlinedButton(onClick = { showConfirm = true }) {
                    Text("Revoke consent & stop processing")
                }
            }
        }
    }

    if (showConfirm) {
        AlertDialog(
            onDismissRequest = { showConfirm = false },
            title = { Text("Revoke biometric consent?") },
            text = {
                Text(
                    "This will:\n" +
                        "• Stop all camera lip-reading immediately.\n" +
                        "• Revoke on-device personalization and permanently delete " +
                        "any stored personalization data.\n\n" +
                        "Liperty will ask for consent again the next time you use it.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    showConfirm = false
                    onRevoke()
                }) { Text("Revoke") }
            },
            dismissButton = {
                TextButton(onClick = { showConfirm = false }) { Text("Cancel") }
            },
        )
    }
}
