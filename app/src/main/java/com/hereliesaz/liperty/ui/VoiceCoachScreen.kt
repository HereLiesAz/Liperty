package com.hereliesaz.liperty.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.hereliesaz.liperty.R
import com.hereliesaz.liperty.voicebox.VoiceViewModel
import com.hereliesaz.liperty.voicebox.cloning.VoiceQualityTier

/**
 * Coached multi-clip voice recording screen.
 *
 * Targets users banking their voice before they lose it (ALS,
 * pre-laryngectomy, vocal-cord surgery, progressive dysarthria).
 * Lets them record short clips in sequence and shows the
 * quality tier ticking up as the embedding centroid stabilizes:
 *
 *     ○○○○  Get started     (no clips yet)
 *     ●○○○  Initial         (1 clip, ≥5 s)
 *     ●●○○  Basic           (3 clips, ≥30 s)
 *     ●●●○  Good            (6 clips, ≥90 s)
 *     ●●●●  Excellent       (12 clips, ≥240 s)
 *
 * Past Excellent, further gains require Tier-3 LoRA fine-tuning.
 *
 * Captured clips are written to cacheDir as 16 kHz WAVs and
 * resampled to 24 kHz at SE-extraction time. They're deleted as
 * soon as the profile is saved or the session is cancelled, so
 * no raw audio outlives the session unless the user explicitly
 * persists it (which we don't offer here).
 */
@Composable
fun VoiceCoachScreen(
    vm: VoiceViewModel = viewModel(),
    onDone: () -> Unit = {},
) {
    val state by vm.coachState.collectAsState()

    // Start a fresh session every time this screen is entered.
    LaunchedEffect(Unit) { vm.startCoachedSession() }

    // Once the save completes, bounce back to the caller.
    LaunchedEffect(state.isSaved) {
        if (state.isSaved) onDone()
    }

    var showSaveDialog by remember { mutableStateOf(false) }
    var showCancelConfirm by remember { mutableStateOf(false) }

    if (showSaveDialog) {
        SaveVoiceDialog(
            defaultName = state.savedProfileName ?: "My Voice",
            onConfirm = { name ->
                showSaveDialog = false
                vm.finishCoachedSession(name)
            },
            onCancel = { showSaveDialog = false }
        )
    }

    if (showCancelConfirm) {
        AlertDialog(
            onDismissRequest = { showCancelConfirm = false },
            title = { Text("Discard recordings?") },
            text = {
                Text(
                    "${state.clipCount} clip(s) will be deleted. " +
                        "Your voice profile won't be saved."
                )
            },
            confirmButton = {
                Button(onClick = {
                    showCancelConfirm = false
                    vm.cancelCoachedSession()
                    onDone()
                }) { Text("Discard") }
            },
            dismissButton = {
                TextButton(onClick = { showCancelConfirm = false }) { Text("Keep recording") }
            },
            containerColor = Color(0xFF12122A),
            titleContentColor = Color.White,
            textContentColor = Color.White,
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0A0A0F))
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "Bank your voice",
            color = Color.White,
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = "Record a few short clips. Each one improves the " +
                "match between the cloned voice and yours.",
            color = Color(0xFFB0B0CC),
            fontSize = 13.sp,
            textAlign = TextAlign.Center,
        )

        // ── Quality tier card ────────────────────────────────────
        QualityTierCard(
            tier = state.qualityTier,
            clipCount = state.clipCount,
            totalSeconds = state.totalSpeechSeconds,
        )

        // ── Script card ──────────────────────────────────────────
        // Tied to promptIndex (not clipCount) so a discard leaves
        // the user on the same prompt they just attempted, ready to
        // retry. The VM advances promptIndex on successful capture
        // only — see VoiceViewModel.recomputeCoachQualityState.
        val script = remember(state.promptIndex) {
            HARVARD_PROMPTS[state.promptIndex % HARVARD_PROMPTS.size]
        }
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF12122A)),
            shape = RoundedCornerShape(16.dp),
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    "Read aloud:",
                    color = Color(0xFF808099),
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = script,
                    color = Color.White,
                    fontSize = 17.sp,
                    fontWeight = FontWeight.Medium,
                )
            }
        }

        // ── Record button ────────────────────────────────────────
        RecordButton(
            isRecording = state.isRecording,
            isSaving = state.isSaving,
            onTap = {
                if (state.isRecording) vm.stopCoachedRecording()
                else vm.recordCoachedClip()
            },
        )

        Text(
            text = state.statusMessage,
            color = if (state.error != null) Color(0xFFFF8A80) else Color(0xFF808099),
            fontSize = 12.sp,
        )

        // ── Captured clips list ─────────────────────────────────
        CapturedClipsList(
            clipCount = state.clipCount,
            onDiscardLast = { vm.discardLastCoachedClip() },
        )

        Spacer(Modifier.height(8.dp))

        // ── Save / cancel row ───────────────────────────────────
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            TextButton(
                onClick = {
                    if (state.clipCount > 0) showCancelConfirm = true else onDone()
                },
                modifier = Modifier.weight(1f),
            ) {
                Text("Cancel", color = Color(0xFFB0B0CC))
            }
            Button(
                onClick = { showSaveDialog = true },
                enabled = state.clipCount > 0 &&
                    !state.isRecording && !state.isSaving,
                modifier = Modifier.weight(2f),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1565C0)),
            ) {
                Icon(Icons.Default.Check, contentDescription = null, tint = Color.White)
                Spacer(Modifier.width(6.dp))
                Text("Save voice profile")
            }
        }
    }
}

@Composable
private fun QualityTierCard(
    tier: VoiceQualityTier,
    clipCount: Int,
    totalSeconds: Int,
) {
    val tierLabel = when (tier) {
        VoiceQualityTier.NONE -> "Get started"
        VoiceQualityTier.INITIAL -> "Initial"
        VoiceQualityTier.BASIC -> "Basic"
        VoiceQualityTier.GOOD -> "Good"
        VoiceQualityTier.EXCELLENT -> "Excellent"
    }
    val tierColor = when (tier) {
        VoiceQualityTier.NONE -> Color(0xFF555570)
        VoiceQualityTier.INITIAL -> Color(0xFFFF9800)
        VoiceQualityTier.BASIC -> Color(0xFFFFEB3B)
        VoiceQualityTier.GOOD -> Color(0xFF8BC34A)
        VoiceQualityTier.EXCELLENT -> Color(0xFF00E5FF)
    }
    val nextTierThreshold = when (tier) {
        VoiceQualityTier.NONE -> "1 clip → Initial"
        VoiceQualityTier.INITIAL -> "3 clips, 30 s → Basic"
        VoiceQualityTier.BASIC -> "6 clips, 90 s → Good"
        VoiceQualityTier.GOOD -> "12 clips, 4 min → Excellent"
        VoiceQualityTier.EXCELLENT -> "Top tier reached. Save when ready."
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF12122A)),
        shape = RoundedCornerShape(16.dp),
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Voice quality",
                    color = Color.White,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                )
                DotIndicator(filled = tier.filledDots, color = tierColor)
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    text = tierLabel,
                    color = tierColor,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "${pluralStringResource(
                        id = R.plurals.voice_coach_clip_count_inline,
                        count = clipCount,
                        clipCount,
                    )} · ${totalSeconds}s",
                    color = Color(0xFF808099),
                    fontSize = 12.sp,
                )
            }
            Text(
                text = "Next: $nextTierThreshold",
                color = Color(0xFF808099),
                fontSize = 11.sp,
            )
        }
    }
}

@Composable
private fun DotIndicator(filled: Int, color: Color, total: Int = 4) {
    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        repeat(total) { i ->
            Box(
                modifier = Modifier
                    .size(14.dp)
                    .clip(CircleShape)
                    .background(
                        if (i < filled) color else Color(0xFF2A2A40)
                    )
            )
        }
    }
}

@Composable
private fun RecordButton(
    isRecording: Boolean,
    isSaving: Boolean,
    onTap: () -> Unit,
) {
    val bg = when {
        isSaving -> Color(0xFF2A2A40)
        isRecording -> Color(0xFFB71C1C)
        else -> Color(0xFF1565C0)
    }
    Button(
        onClick = onTap,
        enabled = !isSaving,
        modifier = Modifier
            .size(96.dp)
            .clip(CircleShape),
        colors = ButtonDefaults.buttonColors(containerColor = bg),
        contentPadding = PaddingValues(0.dp),
    ) {
        if (isSaving) {
            CircularProgressIndicator(color = Color.Cyan, modifier = Modifier.size(36.dp))
        } else {
            Icon(
                imageVector = if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
                contentDescription = if (isRecording) "Stop" else "Record",
                tint = Color.White,
                modifier = Modifier.size(48.dp),
            )
        }
    }
}

@Composable
private fun CapturedClipsList(
    clipCount: Int,
    onDiscardLast: () -> Unit,
) {
    if (clipCount == 0) return
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF0F0F1A)),
        shape = RoundedCornerShape(12.dp),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = pluralStringResource(
                        id = R.plurals.voice_coach_clips_captured,
                        count = clipCount,
                        clipCount,
                    ),
                    color = Color.White,
                    fontSize = 12.sp,
                )
                TextButton(
                    onClick = onDiscardLast,
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp),
                ) {
                    Icon(
                        Icons.Default.Delete,
                        contentDescription = "Discard last clip",
                        tint = Color(0xFFFF8A80),
                        modifier = Modifier.size(16.dp),
                    )
                    Spacer(Modifier.width(4.dp))
                    Text("Discard last", color = Color(0xFFFF8A80), fontSize = 11.sp)
                }
            }
        }
    }
}

@Composable
private fun SaveVoiceDialog(
    defaultName: String,
    onConfirm: (String) -> Unit,
    onCancel: () -> Unit,
) {
    var name by remember { mutableStateOf(defaultName) }
    AlertDialog(
        onDismissRequest = onCancel,
        title = { Text("Name your voice") },
        text = {
            OutlinedTextField(
                value = name,
                onValueChange = { name = it },
                singleLine = true,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White,
                    focusedBorderColor = Color.Cyan,
                    unfocusedBorderColor = Color.Gray,
                    focusedContainerColor = Color(0xFF1A1A2E),
                    unfocusedContainerColor = Color(0xFF0F0F1A),
                ),
            )
        },
        confirmButton = {
            Button(
                onClick = { if (name.isNotBlank()) onConfirm(name) },
                enabled = name.isNotBlank(),
            ) { Text("Save") }
        },
        dismissButton = { TextButton(onClick = onCancel) { Text("Cancel") } },
        containerColor = Color(0xFF12122A),
        titleContentColor = Color.White,
        textContentColor = Color.White,
    )
}

/**
 * Phonetically-balanced reading prompts (Harvard Sentences,
 * IEEE 269 standard test set). Cycle deterministically with
 * clip count so the user covers a wide phonetic distribution
 * even without thinking about what to say.
 */
private val HARVARD_PROMPTS = listOf(
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
)
