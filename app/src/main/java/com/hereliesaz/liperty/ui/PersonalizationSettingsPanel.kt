package com.hereliesaz.liperty.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
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
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * Settings UI for the on-device personalization feature. Surfaces:
 *   - Current consent state (granted / not granted)
 *   - Number of stored paired training records + total bytes
 *   - "Train personal model" button (placeholder until Step 2/3 wire up)
 *   - "Revoke consent + delete all data" with confirmation
 *
 * The panel is self-contained — it takes only the values it needs to
 * render and the callbacks it needs to invoke. Mounting this in
 * SettingsActivity (or a dedicated PersonalizationSettingsActivity) is
 * the caller's job.
 *
 * State is passed in via plain Kotlin values (count, bytes, consent
 * granted). The owning Activity reads from
 * [com.hereliesaz.liperty.personalization.PairedTrainingStore] and
 * [com.hereliesaz.liperty.personalization.PersonalizationConsentManager]
 * and passes the result here. Keeps this composable independent of
 * Android `Context` for previews and tests.
 */
@Composable
fun PersonalizationSettingsPanel(
    consentGranted: Boolean,
    consentGrantedAtMs: Long?,
    recordCount: Int,
    totalBytes: Long,
    onGrantConsent: () -> Unit,
    onRevokeConsent: () -> Unit,
    onDeleteAll: () -> Unit,
    onTrainNow: (() -> Unit)? = null,
) {
    var showRevokeConfirm by remember { mutableStateOf(false) }
    var showDeleteConfirm by remember { mutableStateOf(false) }

    Card(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                "Personal lipreading model",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Spacer(Modifier.height(8.dp))

            if (consentGranted) {
                Text(
                    "On — Liperty is collecting paired video/audio data to personalize " +
                        "the lipreading model for you. Everything stays on this device.",
                    style = MaterialTheme.typography.bodySmall,
                )
                consentGrantedAtMs?.let { ms ->
                    Spacer(Modifier.height(4.dp))
                    Text(
                        "Enabled on ${formatTimestamp(ms)}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            } else {
                Text(
                    "Off — Liperty is using the standard lipreading model.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            Spacer(Modifier.height(12.dp))
            Text(
                "Stored data: $recordCount samples, ${formatBytes(totalBytes)}",
                style = MaterialTheme.typography.bodySmall,
            )

            Spacer(Modifier.height(16.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (!consentGranted) {
                    OutlinedButton(onClick = onGrantConsent) { Text("Enable") }
                } else {
                    if (onTrainNow != null && recordCount > 0) {
                        OutlinedButton(onClick = onTrainNow) { Text("Train now") }
                    }
                    if (recordCount > 0) {
                        OutlinedButton(onClick = { showDeleteConfirm = true }) {
                            Text("Delete stored data")
                        }
                    }
                    OutlinedButton(onClick = { showRevokeConfirm = true }) {
                        Text("Revoke consent")
                    }
                }
            }
        }
    }

    if (showRevokeConfirm) {
        AlertDialog(
            onDismissRequest = { showRevokeConfirm = false },
            title = { Text("Revoke personalization consent?") },
            text = {
                Text(
                    "This will:\n" +
                        "• Stop collecting paired video/audio from voice recordings.\n" +
                        "• Permanently delete all stored personalization data " +
                        "(${recordCount} samples, ${formatBytes(totalBytes)}).\n" +
                        "• Revert to the standard lipreading model.\n\n" +
                        "This cannot be undone.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    showRevokeConfirm = false
                    onRevokeConsent()
                }) { Text("Revoke and delete") }
            },
            dismissButton = {
                TextButton(onClick = { showRevokeConfirm = false }) { Text("Cancel") }
            },
        )
    }

    if (showDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = false },
            title = { Text("Delete stored data?") },
            text = {
                Text(
                    "Permanently delete the ${recordCount} stored samples (${formatBytes(totalBytes)})? " +
                        "Personalization consent remains active and Liperty will collect new data " +
                        "on your next voice recording session.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    showDeleteConfirm = false
                    onDeleteAll()
                }) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = false }) { Text("Cancel") }
            },
        )
    }
}

private fun formatBytes(b: Long): String = when {
    b < 1_024 -> "$b B"
    b < 1_048_576 -> "${b / 1_024} KB"
    b < 1_073_741_824 -> "${b / 1_048_576} MB"
    else -> "${"%.1f".format(b / 1_073_741_824.0)} GB"
}

private fun formatTimestamp(ms: Long): String =
    java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US)
        .format(java.util.Date(ms))
