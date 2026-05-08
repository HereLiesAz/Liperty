package com.hereliesaz.liperty.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.hereliesaz.liperty.voicebox.ImportState
import com.hereliesaz.liperty.voicebox.ImportStep
import com.hereliesaz.liperty.voicebox.VoiceViewModel
import com.hereliesaz.liperty.voicebox.cloning.SpeakerClusterer
import com.hereliesaz.liperty.voicebox.cloning.VoiceProfile

/**
 * Multi-step import wizard for voice cloning from uncontrolled audio/video files.
 *
 * Steps:
 * 1. File Selection
 * 2. Processing (extraction, VAD, clustering)
 * 3. Speaker Identification (if multiple speakers detected)
 * 4. Profile Configuration (name, quality preview, add-to-existing)
 * 5. Confirmation
 */
@Composable
fun VoiceImportWizardScreen(
    vm: VoiceViewModel,
    onDismiss: () -> Unit
) {
    val importState by vm.importState.collectAsState()
    val uiState by vm.uiState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0A0A0F))
            .padding(24.dp)
    ) {
        // Top bar with back button and step indicator
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            IconButton(onClick = onDismiss) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back", tint = Color.White)
            }
            Text(
                stepTitle(importState.step),
                color = Color.White,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f)
            )
        }

        Spacer(Modifier.height(16.dp))

        when (importState.step) {
            ImportStep.IDLE -> FileSelectionStep(vm)
            ImportStep.EXTRACTING, ImportStep.SEGMENTING, ImportStep.CLUSTERING ->
                ProcessingStep(importState)
            ImportStep.SPEAKER_SELECTION -> SpeakerIdentificationStep(importState, vm)
            ImportStep.READY -> ProfileConfigurationStep(importState, uiState.profiles, vm)
            ImportStep.SAVING -> SavingStep()
            ImportStep.COMPLETE -> ConfirmationStep(onDismiss)
            ImportStep.ERROR -> ErrorStep(importState, vm, onDismiss)
        }
    }
}

private fun stepTitle(step: ImportStep) = when (step) {
    ImportStep.IDLE -> "Import Recordings"
    ImportStep.EXTRACTING -> "Processing..."
    ImportStep.SEGMENTING -> "Detecting Speech..."
    ImportStep.CLUSTERING -> "Analyzing Speakers..."
    ImportStep.SPEAKER_SELECTION -> "Identify Your Voice"
    ImportStep.READY -> "Configure Profile"
    ImportStep.SAVING -> "Saving..."
    ImportStep.COMPLETE -> "Done"
    ImportStep.ERROR -> "Error"
}

// ── Step 1: File Selection ──────────────────────────────────────────────

@Composable
private fun FileSelectionStep(vm: VoiceViewModel) {
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetMultipleContents()
    ) { uris ->
        if (uris.isNotEmpty()) {
            vm.startImportProcessing(uris)
        }
    }

    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            Icons.Default.AudioFile,
            contentDescription = null,
            tint = Color.Cyan,
            modifier = Modifier.size(80.dp)
        )

        Spacer(Modifier.height(24.dp))

        Text(
            "Select audio or video files containing your voice.",
            color = Color(0xFFB0B0CC),
            fontSize = 16.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 32.dp)
        )

        Spacer(Modifier.height(8.dp))

        Text(
            "Home videos, voicemails, voice memos — anything works.\nMultiple speakers are handled automatically.",
            color = Color(0xFF808099),
            fontSize = 13.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 32.dp)
        )

        Spacer(Modifier.height(32.dp))

        Button(
            onClick = { filePickerLauncher.launch("audio/*") },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 32.dp)
                .height(56.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1565C0)),
            shape = RoundedCornerShape(16.dp)
        ) {
            Icon(Icons.Default.FileUpload, contentDescription = null)
            Spacer(Modifier.width(12.dp))
            Text("Select Audio Files", fontSize = 16.sp)
        }

        Spacer(Modifier.height(12.dp))

        OutlinedButton(
            onClick = { filePickerLauncher.launch("video/*") },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 32.dp)
                .height(56.dp),
            shape = RoundedCornerShape(16.dp)
        ) {
            Icon(Icons.Default.VideoFile, contentDescription = null, tint = Color.Cyan)
            Spacer(Modifier.width(12.dp))
            Text("Select Video Files", color = Color.Cyan, fontSize = 16.sp)
        }
    }
}

// ── Step 2: Processing ──────────────────────────────────────────────────

@Composable
private fun ProcessingStep(state: ImportState) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        CircularProgressIndicator(
            color = Color.Cyan,
            modifier = Modifier.size(64.dp),
            strokeWidth = 4.dp
        )

        Spacer(Modifier.height(24.dp))

        Text(state.progressMessage, color = Color.White, fontSize = 16.sp)

        if (state.progressPercent > 0f) {
            Spacer(Modifier.height(16.dp))
            LinearProgressIndicator(
                progress = { state.progressPercent },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 48.dp)
                    .height(6.dp)
                    .clip(RoundedCornerShape(3.dp)),
                color = Color.Cyan,
                trackColor = Color(0xFF1A1A2E),
            )
        }
    }
}

// ── Step 3: Speaker Identification ──────────────────────────────────────

@Composable
private fun SpeakerIdentificationStep(state: ImportState, vm: VoiceViewModel) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            "Multiple speakers were detected.\nWhich voice is yours?",
            color = Color(0xFFB0B0CC),
            fontSize = 14.sp,
            textAlign = TextAlign.Center
        )

        Spacer(Modifier.height(20.dp))

        LazyColumn(
            verticalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        ) {
            items(state.clusters) { cluster ->
                SpeakerCard(
                    cluster = cluster,
                    isSelected = state.selectedClusterId == cluster.clusterId,
                    onPlay = { vm.playSegmentPreview(cluster.representativeSegment) },
                    onSelect = { vm.selectSpeaker(cluster.clusterId) }
                )
            }
        }

        Spacer(Modifier.height(16.dp))

        Button(
            onClick = { vm.confirmSpeakerSelection() },
            enabled = state.selectedClusterId != null,
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1565C0)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Text("Continue", fontSize = 16.sp)
        }
    }
}

@Composable
private fun SpeakerCard(
    cluster: SpeakerClusterer.SpeakerCluster,
    isSelected: Boolean,
    onPlay: () -> Unit,
    onSelect: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onSelect),
        colors = CardDefaults.cardColors(
            containerColor = if (isSelected) Color(0xFF1A2A3E) else Color(0xFF12121F)
        ),
        shape = RoundedCornerShape(12.dp),
        border = if (isSelected) CardDefaults.outlinedCardBorder() else null
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Play button
            IconButton(
                onClick = onPlay,
                modifier = Modifier
                    .size(48.dp)
                    .clip(CircleShape)
                    .background(Color(0xFF1565C0))
            ) {
                Icon(Icons.Default.PlayArrow, "Play sample", tint = Color.White)
            }

            Spacer(Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    "Speaker ${cluster.clusterId + 1}",
                    color = Color.White,
                    fontWeight = FontWeight.Medium
                )
                Text(
                    "${cluster.segments.size} segments · ${cluster.totalSpeechDurationMs / 1000}s",
                    color = Color(0xFF808099),
                    fontSize = 12.sp
                )
            }

            if (isSelected) {
                Icon(Icons.Default.CheckCircle, "Selected", tint = Color.Cyan)
            }
        }
    }
}

// ── Step 4: Profile Configuration ───────────────────────────────────────

@Composable
private fun ProfileConfigurationStep(
    state: ImportState,
    existingProfiles: List<VoiceProfile>,
    vm: VoiceViewModel
) {
    var name by remember { mutableStateOf("") }
    var addToExisting by remember { mutableStateOf(false) }
    var selectedExisting by remember { mutableStateOf<String?>(null) }

    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Quality indicator
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF12122A)),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text("Voice Quality", color = Color(0xFFB0B0CC), fontSize = 12.sp)
                LinearProgressIndicator(
                    progress = { state.qualityScore },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp)),
                    color = qualityColor(state.qualityScore),
                    trackColor = Color(0xFF1A1A2E),
                )
                Text(
                    qualityLabel(state.qualityScore),
                    color = qualityColor(state.qualityScore),
                    fontWeight = FontWeight.Bold
                )
                Text(
                    "${state.sampleCount} segments · ${state.totalSpeechDurationMs / 1000}s speech",
                    color = Color(0xFF808099),
                    fontSize = 12.sp
                )
            }
        }

        // Mode toggle: new profile vs. add to existing
        if (existingProfiles.isNotEmpty()) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                FilterChip(
                    selected = !addToExisting,
                    onClick = { addToExisting = false },
                    label = { Text("New Profile") }
                )
                FilterChip(
                    selected = addToExisting,
                    onClick = { addToExisting = true },
                    label = { Text("Add to Existing") }
                )
            }
        }

        if (addToExisting) {
            // Dropdown of existing profiles
            LazyColumn(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(existingProfiles) { profile ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { selectedExisting = profile.name },
                        colors = CardDefaults.cardColors(
                            containerColor = if (selectedExisting == profile.name) Color(0xFF1A2A3E)
                            else Color(0xFF0F0F1A)
                        ),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(profile.name, color = Color.White, modifier = Modifier.weight(1f))
                            Text(
                                "${profile.sampleCount} samples",
                                color = Color(0xFF808099),
                                fontSize = 12.sp
                            )
                            if (selectedExisting == profile.name) {
                                Spacer(Modifier.width(8.dp))
                                Icon(Icons.Default.Check, "Selected", tint = Color.Cyan, modifier = Modifier.size(20.dp))
                            }
                        }
                    }
                }
            }
        } else {
            // Name input
            OutlinedTextField(
                value = name,
                onValueChange = { name = it },
                placeholder = { Text("e.g. My Old Voice") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White,
                    focusedBorderColor = Color.Cyan,
                    unfocusedBorderColor = Color.Gray,
                    focusedContainerColor = Color(0xFF1A1A2E),
                    unfocusedContainerColor = Color(0xFF0F0F1A)
                )
            )
            Spacer(Modifier.weight(1f))
        }

        // Preview button
        OutlinedButton(
            onClick = { vm.previewVoice() },
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(Icons.Default.VolumeUp, contentDescription = null, tint = Color.Cyan)
            Spacer(Modifier.width(8.dp))
            Text("Preview Voice", color = Color.Cyan)
        }

        // Save button
        val canSave = if (addToExisting) selectedExisting != null else name.isNotBlank()
        Button(
            onClick = {
                if (addToExisting) {
                    vm.saveImportedProfile(selectedExisting!!, selectedExisting)
                } else {
                    vm.saveImportedProfile(name, null)
                }
            },
            enabled = canSave,
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1565C0)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Text(if (addToExisting) "Add Samples" else "Create Voice Profile", fontSize = 16.sp)
        }
    }
}

// ── Step 5+: Terminal states ────────────────────────────────────────────

@Composable
private fun SavingStep() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CircularProgressIndicator(color = Color.Cyan)
            Spacer(Modifier.height(16.dp))
            Text("Saving voice profile...", color = Color.White)
        }
    }
}

@Composable
private fun ConfirmationStep(onDismiss: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(Icons.Default.CheckCircle, null, tint = Color(0xFF4CAF50), modifier = Modifier.size(80.dp))
        Spacer(Modifier.height(24.dp))
        Text("Voice profile saved!", color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(8.dp))
        Text("You can add more samples anytime to improve quality.", color = Color(0xFFB0B0CC), fontSize = 14.sp)
        Spacer(Modifier.height(32.dp))
        Button(
            onClick = onDismiss,
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1565C0)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Text("Done")
        }
    }
}

@Composable
private fun ErrorStep(state: ImportState, vm: VoiceViewModel, onDismiss: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(Icons.Default.Error, null, tint = Color(0xFFFF5252), modifier = Modifier.size(64.dp))
        Spacer(Modifier.height(16.dp))
        Text(state.error ?: "Something went wrong", color = Color.White, fontSize = 16.sp)
        Spacer(Modifier.height(24.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = onDismiss) { Text("Cancel", color = Color.White) }
            Button(
                onClick = { vm.resetImport() },
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1565C0))
            ) {
                Text("Try Again")
            }
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

private fun qualityColor(score: Float) = when {
    score >= 0.7f -> Color(0xFF4CAF50) // Green
    score >= 0.4f -> Color(0xFFFFEB3B) // Yellow
    else -> Color(0xFFFF9800)          // Orange
}

private fun qualityLabel(score: Float) = when {
    score >= 0.8f -> "Excellent"
    score >= 0.6f -> "Good"
    score >= 0.4f -> "Fair"
    score >= 0.2f -> "Basic"
    else -> "Needs more samples"
}
