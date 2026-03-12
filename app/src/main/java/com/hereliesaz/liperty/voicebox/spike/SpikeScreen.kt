package com.hereliesaz.liperty.voicebox.spike

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel

/**
 * Phase 0 Spike Screen.
 *
 * Composable screen wired into the main AzNavHost via the "voicebox" route.
 * Replaces the old standalone SpikeActivity — no manifest entry needed.
 *
 * Records 10 seconds of dual-stream (contact-mic + accelerometer) data to
 * the app's external files directory for analysis with analyze.py.
 */
@Composable
fun SpikeScreen(
    vm: SpikeViewModel = viewModel()
) {
    val state by vm.uiState.collectAsState()
    val context = LocalContext.current

    // Permission state — check on every recomposition in case user grants mid-session
    var hasAudioPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasAudioPermission = granted }

    // Animate vibration bar width (0f–1f)
    val barFraction by animateFloatAsState(
        targetValue   = (state.accelMagnitude * 20f).coerceIn(0f, 1f),
        animationSpec = tween(durationMillis = 80),
        label         = "vibration_bar"
    )
    val barColor = when {
        barFraction > 0.6f -> Color(0xFF00E676)   // Green  — strong signal
        barFraction > 0.2f -> Color(0xFFFFD740)   // Amber  — weak signal
        else               -> Color(0xFF37474F)   // Dark   — no signal
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0A0A0F))
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(18.dp)
    ) {

        // Title
        Text(
            text       = "Voice Box · Signal Spike",
            color      = Color.White,
            fontSize   = 22.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text     = if (state.isLiveListening) "Phase 1 — Artificial Larynx SSI" else "Phase 0 — Signal Feasibility",
            color    = if (state.isLiveListening) Color.Cyan else Color(0xFF666680),
            fontSize = 13.sp
        )

        HorizontalDivider(color = Color(0xFF1E1E2E))

        // Live Mode Toggle
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Larynx Vibration", color = Color.White, fontSize = 14.sp)
            Switch(
                checked = state.isLiveListening,
                onCheckedChange = { 
                    if (hasAudioPermission) vm.toggleLiveListen() 
                    else permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                },
                colors = SwitchDefaults.colors(checkedThumbColor = Color.Cyan)
            )
        }

        if (state.isLiveListening) {
            // Real-time meters
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                MeterBar(label = "CONTACT PRESSURE / COARTICULATION", value = state.accelMagnitude * 20f, color = if (state.isVoicing) Color(0xFF00E676) else Color(0xFF37474F))
                MeterBar(label = "RECONSTRUCTED SSI SIGNAL", value = state.audioLevel, color = Color.Cyan)
            }
        }

        // Placement guide card
        Card(
            shape  = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF12122A))
        ) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("📱  SSI Placement", color = Color(0xFF7EB6FF), fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
                Text(
                    text = "Press the back of the device firmly against your\n" +
                            "throat below the Adam's apple. The phone will\n" +
                            "vibrate to provide a sound source.\n\n" +
                            "Mouth words silently. Your articulators will\n" +
                            "modulate the vibration into speech signals.",
                    color      = Color(0xFFB0B0CC),
                    fontSize   = 13.sp,
                    lineHeight = 20.sp
                )
            }
        }

        // Vibration indicator
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(
                text          = "LARYNGEAL VIBRATION",
                color         = Color(0xFF555570),
                fontSize      = 10.sp,
                letterSpacing = 2.sp
            )
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(20.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(Color(0xFF1A1A2E))
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(barFraction)
                        .fillMaxHeight()
                        .clip(RoundedCornerShape(10.dp))
                        .background(barColor)
                )
            }
            Text(
                text       = "%.3f m/s²".format(state.accelMagnitude),
                color      = Color(0xFF444460),
                fontSize   = 10.sp,
                fontFamily = FontFamily.Monospace
            )
        }

        // Status line
        Text(
            text      = state.statusText,
            color     = if (state.isRecording) Color(0xFFFF5252) else Color(0xFF9090AA),
            fontSize  = 13.sp,
            textAlign = TextAlign.Center
        )

        // Error (if any)
        if (state.errorText.isNotEmpty()) {
            Text(
                text      = state.errorText,
                color     = Color(0xFFFF6E6E),
                fontSize  = 12.sp,
                textAlign = TextAlign.Center,
                fontFamily = FontFamily.Monospace
            )
        }

        // Record / Stop button
        if (!hasAudioPermission) {
            Button(
                onClick = { permissionLauncher.launch(Manifest.permission.RECORD_AUDIO) },
                modifier = Modifier
                    .size(130.dp)
                    .clip(CircleShape),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF37474F))
            ) {
                Text(
                    text      = "Grant\nMic",
                    textAlign = TextAlign.Center,
                    fontSize  = 14.sp
                )
            }
        } else {
            Button(
                onClick = {
                    if (state.isRecording) vm.stopRecording() else vm.startRecording()
                },
                modifier = Modifier
                    .size(130.dp)
                    .clip(CircleShape),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (state.isRecording) Color(0xFFB71C1C) else Color(0xFF1565C0)
                )
            ) {
                Text(
                    text       = if (state.isRecording) "STOP" else "RECORD",
                    fontSize   = 16.sp,
                    fontWeight = FontWeight.Bold,
                    textAlign  = TextAlign.Center
                )
            }
        }

        // Results card
        if (state.resultLines.isNotEmpty()) {
            Card(
                shape  = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF06111F))
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    state.resultLines.forEach { line ->
                        Text(
                            text       = line,
                            color      = if (line.startsWith("✅")) Color(0xFF80CBC4)
                            else Color(0xFF607D8B),
                            fontSize   = 11.sp,
                            fontFamily = FontFamily.Monospace,
                            lineHeight = 17.sp
                        )
                    }
                }
            }
        }

        Spacer(Modifier.height(16.dp))
    }
}

@Composable
private fun MeterBar(label: String, value: Float, color: Color) {
    val animatedValue by animateFloatAsState(
        targetValue = value.coerceIn(0f, 1f),
        animationSpec = tween(durationMillis = 80),
        label = "meter_anim"
    )
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(text = label, color = Color(0xFF555570), fontSize = 10.sp, letterSpacing = 1.sp)
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(12.dp)
                .clip(RoundedCornerShape(6.dp))
                .background(Color(0xFF1A1A2E))
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(animatedValue)
                    .fillMaxHeight()
                    .background(color)
            )
        }
    }
}
