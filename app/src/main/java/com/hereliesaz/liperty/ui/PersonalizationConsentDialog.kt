package com.hereliesaz.liperty.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * Explicit opt-in dialog for the on-device personalization feature.
 * Distinct from the app-launch consent gate — this is the second,
 * narrower consent specifically for persisting biometric data
 * (lip-cropped video + audio + optional transcript) on-device.
 *
 * Show this when the user first activates a feature that would create
 * a [com.hereliesaz.liperty.personalization.PairedTrainingRecord]
 * (currently: importing video into the voice-cloning flow with the
 * personalization feature enabled). The result feeds
 * [com.hereliesaz.liperty.personalization.PersonalizationConsentManager.grantConsent]
 * or simply does nothing on decline.
 *
 * Plain text rather than a marketing-style explainer — the user needs
 * to actually understand what's being stored and where.
 */
@Composable
fun PersonalizationConsentDialog(
    onAccept: () -> Unit,
    onDecline: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDecline,
        title = {
            Text(
                "Personalize lipreading for you",
                fontWeight = FontWeight.Bold,
            )
        },
        text = {
            Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
                Text(
                    "Liperty's lipreading models are trained on a general population. " +
                        "If your face, mouth, or speaking style differs from that population — " +
                        "as is common for Deaf and Hard-of-Hearing speakers, people with speech " +
                        "impairments, or speakers across the global demographic range — accuracy " +
                        "may be lower for you.",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Spacer(Modifier.height(12.dp))
                Text(
                    "What this opts you into:",
                    fontWeight = FontWeight.SemiBold,
                    style = MaterialTheme.typography.bodyMedium,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    "• Liperty stores cropped lip-area video from your voice recordings on this device.\n" +
                        "• It stores the audio from those recordings.\n" +
                        "• It optionally transcribes the audio and stores the transcript.\n" +
                        "• This data is used to personalize the lipreading model FOR YOU on this device.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Spacer(Modifier.height(12.dp))
                Text(
                    "What this does NOT do:",
                    fontWeight = FontWeight.SemiBold,
                    style = MaterialTheme.typography.bodyMedium,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    "• It does not upload anything off your device.\n" +
                        "• It does not store your face, only the cropped lip area.\n" +
                        "• It does not run continuously — only during voice imports/recordings.\n" +
                        "• It does not share data with other Liperty users.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Spacer(Modifier.height(12.dp))
                Text(
                    "You can revoke this consent at any time in Settings. " +
                        "Revoking permanently deletes all stored personalization data.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        },
        confirmButton = {
            TextButton(onClick = onAccept) {
                Text("I agree — personalize for me")
            }
        },
        dismissButton = {
            TextButton(onClick = onDecline) {
                Text("No thanks")
            }
        },
    )
}
