package com.hereliesaz.liperty.ui

import android.graphics.Bitmap
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.hereliesaz.liperty.ml.CalibrationViewModel
import com.hereliesaz.liperty.ui.CalibrationScreen
import com.hereliesaz.aznavrail.AzHostActivityLayout
import com.hereliesaz.aznavrail.AzNavHost

@Composable
fun LipertyApp(
    previewView: PreviewView,
    overlayView: OverlayView,
    transcriptionWords: List<String>,
    wordConfidences: List<Float> = emptyList(),
    selectedWordIndex: Int,
    onWordClick: (Int) -> Unit,
    isRecording: Boolean,
    onSwitchCamera: () -> Unit,
    onOpenSettings: () -> Unit,
    onClearTranscript: () -> Unit,
    onSpeak: () -> Unit,
    onToggleSSI: () -> Unit = {},
    onToggleLipRead: () -> Unit = {},
    onToggleEL: () -> Unit = {},
    isPaused: Boolean = false,
    isSSIActive: Boolean = false,
    isLipReadActive: Boolean = false,
    isELActive: Boolean = false,
    currentLensFacing: Int = 0, // 0 for Front, 1 for Back
    vsrSensitivity: Float = 0.5f,
    onVsrSensitivityChange: (Float) -> Unit = {},
    larynxSensitivity: Float = 0.5f,
    onLarynxSensitivityChange: (Float) -> Unit = {},
    isDarkTheme: Boolean = true,
    onRegisterCalibrationCallback: (((Bitmap) -> Unit)?) -> Unit = {}
) {
    val navController = rememberNavController()

    LipertyTheme {
    val backgroundColor = MaterialTheme.colorScheme.background

    Box(modifier = Modifier.fillMaxSize().background(backgroundColor)) {

        // Camera preview + overlay in the background
        AndroidView(
            factory = { ctx ->
                FrameLayout(ctx).apply {
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    if (previewView.parent != null)
                        (previewView.parent as ViewGroup).removeView(previewView)
                    addView(previewView)

                    if (overlayView.parent != null)
                        (overlayView.parent as ViewGroup).removeView(overlayView)
                    addView(overlayView)
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        AzHostActivityLayout(
            navController      = navController,
            initiallyExpanded  = false
        ) {
            azConfig(
                packButtons    = true,
                displayAppName = true
            )

            azTheme(activeColor = Color.Cyan)

            // ── Navigation rail items ─────────────────────────────────────
            azRailItem(
                id      = "home",
                text    = "Home",
                route   = "home",
                content = Icons.Filled.Home,
                color   = Color.White
            )

            azRailToggle(
                id            = "lipread",
                isChecked     = isLipReadActive,
                toggleOnText  = "Lip-Read ON",
                toggleOffText = "Lip-Read OFF",
                onClick       = { onToggleLipRead() },
                color         = Color.White
            )

            azRailToggle(
                id            = "voicebox",
                isChecked     = isSSIActive,
                toggleOnText  = "Larynx ON",
                toggleOffText = "Larynx OFF",
                onClick       = { onToggleSSI() },
                color         = Color.White
            )
            
            azRailToggle(
                id            = "el_translator",
                isChecked     = isELActive,
                toggleOnText  = "EL-Transl ON",
                toggleOffText = "EL-Transl OFF",
                onClick       = { onToggleEL() },
                color         = Color.Magenta
            )

            azRailToggle(
                id            = "switch_cam",
                isChecked     = currentLensFacing == 1, // 1 for Back
                toggleOnText  = "Back",
                toggleOffText = "Front",
                onClick       = { onSwitchCamera() },
                color         = Color.White
            )

            azRailItem(
                id      = "voice_mgmt",
                text    = "Voice",
                route   = "voice_mgmt",
                content = Icons.Filled.RecordVoiceOver,
                color   = Color.White
            )

            azRailItem(
                id      = "calibrate",
                text    = "Tweak",
                route   = "calibrate",
                content = Icons.Filled.Refresh,
                color   = Color.White
            )

            azRailItem(
                id      = "settings",
                text    = "Settings",
                route   = "settings",
                content = Icons.Filled.Settings,
                color   = Color.White
            )

            // ── Action menu items ─────────────────────────────────────────
            azMenuItem(
                id      = "clear",
                text    = "Clear Transcript",
                route   = "clear",
                content = Icons.Filled.Clear
            )

            azMenuItem(
                id      = "speak",
                text    = "Speak Text",
                route   = "speak",
                content = Icons.Filled.PlayArrow
            )

            // ── Screen content ────────────────────────────────────────────
            onscreen(alignment = Alignment.Center) {
                var fontSize by remember { mutableFloatStateOf(24f) }
                val transformState = rememberTransformableState { zoomChange, _, _ ->
                    fontSize = (fontSize * zoomChange).coerceIn(12f, 120f)
                }

                AzNavHost(navController = navController, startDestination = "home") {

                    composable("home") {
                        val anyModeActive = isLipReadActive || isSSIActive || isELActive

                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .transformable(state = transformState)
                        ) {
                            // ── Status badge ──────────────────────────────
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 16.dp, vertical = 8.dp),
                                horizontalArrangement = Arrangement.End,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                if (isRecording) {
                                    val (badgeText, badgeColor) = when {
                                        isPaused    -> "PAUSED"  to Color(0xFFFFD600)
                                        isELActive  -> "EL"      to MaterialTheme.colorScheme.tertiary
                                        isSSIActive -> "SSI"     to MaterialTheme.colorScheme.primary
                                        isLipReadActive -> "READING" to Color(0xFFFF5252)
                                        else        -> "REC"     to Color(0xFFFF5252)
                                    }
                                    Box(
                                        modifier = Modifier
                                            .clip(RoundedCornerShape(50))
                                            .background(badgeColor.copy(alpha = 0.18f))
                                            .padding(horizontal = 12.dp, vertical = 4.dp)
                                    ) {
                                        Text(
                                            text = badgeText,
                                            color = badgeColor,
                                            style = MaterialTheme.typography.labelMedium,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                }
                            }

                            // ── Transcription area ────────────────────────
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .weight(1f),
                                contentAlignment = Alignment.Center
                            ) {
                                when {
                                    isLipReadActive && transcriptionWords.isNotEmpty() -> {
                                        FlowRow(
                                            modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp),
                                            horizontalArrangement = Arrangement.Center,
                                            verticalArrangement = Arrangement.Center
                                        ) {
                                            transcriptionWords.forEachIndexed { index, word ->
                                                val isSelected = index == selectedWordIndex
                                                val confidence = wordConfidences.getOrElse(index) { 0.5f }
                                                val heatColor = lerp(
                                                    lerp(Color(0xFFB71C1C), Color(0xFFF57F17), confidence * 2f),
                                                    lerp(Color(0xFFF57F17), Color(0xFF1B5E20), (confidence - 0.5f) * 2f),
                                                    if (confidence < 0.5f) 0f else 1f
                                                ).copy(alpha = 0.25f)
                                                Text(
                                                    text = word,
                                                    color = if (isSelected)
                                                        MaterialTheme.colorScheme.primary
                                                    else
                                                        MaterialTheme.colorScheme.onBackground,
                                                    fontSize = fontSize.sp,
                                                    fontWeight = if (isSelected) FontWeight.ExtraBold else FontWeight.Normal,
                                                    modifier = Modifier
                                                        .padding(4.dp)
                                                        .clip(RoundedCornerShape(8.dp))
                                                        .background(
                                                            if (isSelected)
                                                                MaterialTheme.colorScheme.primary.copy(alpha = 0.18f)
                                                            else
                                                                heatColor
                                                        )
                                                        .clickable { onWordClick(index) }
                                                        .padding(horizontal = 6.dp, vertical = 2.dp)
                                                )
                                            }
                                        }
                                    }

                                    // A mode is active but no text has arrived yet
                                    anyModeActive -> {
                                        val infiniteTransition = rememberInfiniteTransition(label = "pulse")
                                        val alpha by infiniteTransition.animateFloat(
                                            initialValue = 0.3f,
                                            targetValue  = 1f,
                                            animationSpec = infiniteRepeatable(
                                                animation  = tween(900, easing = LinearEasing),
                                                repeatMode = RepeatMode.Reverse
                                            ),
                                            label = "pulseAlpha"
                                        )
                                        Column(
                                            horizontalAlignment = Alignment.CenterHorizontally,
                                            verticalArrangement = Arrangement.spacedBy(16.dp)
                                        ) {
                                            LinearProgressIndicator(
                                                modifier = Modifier
                                                    .fillMaxWidth(0.55f)
                                                    .clip(RoundedCornerShape(50)),
                                                color = when {
                                                    isELActive  -> MaterialTheme.colorScheme.tertiary
                                                    isSSIActive -> MaterialTheme.colorScheme.primary
                                                    else        -> MaterialTheme.colorScheme.primary
                                                },
                                                trackColor = MaterialTheme.colorScheme.surfaceContainerHigh
                                            )
                                            Text(
                                                text = when {
                                                    isELActive  -> "Listening for electrolarynx…"
                                                    isSSIActive -> "Listening for throat vibrations…"
                                                    else        -> "Watching for lip movements…"
                                                },
                                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = alpha),
                                                style = MaterialTheme.typography.bodyLarge,
                                                textAlign = TextAlign.Center
                                            )
                                        }
                                    }

                                    // Nothing is active
                                    else -> {
                                        Text(
                                            text = "Enable a mode from the\nnavigation rail to start",
                                            color = MaterialTheme.colorScheme.outlineVariant,
                                            style = MaterialTheme.typography.bodyMedium,
                                            textAlign = TextAlign.Center
                                        )
                                    }
                                }
                            }

                            // ── Sensitivity slider(s) docked to bottom ────
                            if (anyModeActive) {
                                Surface(
                                    modifier = Modifier.fillMaxWidth(),
                                    shape = RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp),
                                    color = MaterialTheme.colorScheme.surfaceContainerHigh,
                                    tonalElevation = 4.dp
                                ) {
                                    Column(
                                        modifier = Modifier.padding(
                                            horizontal = 20.dp,
                                            vertical = 12.dp
                                        ),
                                        verticalArrangement = Arrangement.spacedBy(4.dp)
                                    ) {
                                        if (isLipReadActive) {
                                            Row(
                                                modifier = Modifier.fillMaxWidth(),
                                                horizontalArrangement = Arrangement.SpaceBetween,
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Text(
                                                    text = "Detection Sensitivity",
                                                    style = MaterialTheme.typography.labelLarge,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                                Text(
                                                    text = "${(vsrSensitivity * 100).toInt()}%",
                                                    style = MaterialTheme.typography.labelLarge,
                                                    fontWeight = FontWeight.Bold,
                                                    color = MaterialTheme.colorScheme.primary
                                                )
                                            }
                                            Slider(
                                                value = vsrSensitivity,
                                                onValueChange = onVsrSensitivityChange,
                                                modifier = Modifier.fillMaxWidth(),
                                                colors = SliderDefaults.colors(
                                                    thumbColor       = MaterialTheme.colorScheme.primary,
                                                    activeTrackColor = MaterialTheme.colorScheme.primary,
                                                    inactiveTrackColor = MaterialTheme.colorScheme.outline
                                                )
                                            )
                                        }

                                        if (isSSIActive || isELActive) {
                                            if (isLipReadActive) Spacer(Modifier.height(4.dp))
                                            val sliderColor = if (isELActive)
                                                MaterialTheme.colorScheme.tertiary
                                            else
                                                MaterialTheme.colorScheme.primary
                                            val sliderLabel = if (isELActive)
                                                "EL Voice Activity Sensitivity"
                                            else
                                                "Larynx Sensitivity"
                                            Row(
                                                modifier = Modifier.fillMaxWidth(),
                                                horizontalArrangement = Arrangement.SpaceBetween,
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Text(
                                                    text = sliderLabel,
                                                    style = MaterialTheme.typography.labelLarge,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                                Text(
                                                    text = "${(larynxSensitivity * 100).toInt()}%",
                                                    style = MaterialTheme.typography.labelLarge,
                                                    fontWeight = FontWeight.Bold,
                                                    color = sliderColor
                                                )
                                            }
                                            Slider(
                                                value = larynxSensitivity,
                                                onValueChange = onLarynxSensitivityChange,
                                                modifier = Modifier.fillMaxWidth(),
                                                colors = SliderDefaults.colors(
                                                    thumbColor         = sliderColor,
                                                    activeTrackColor   = sliderColor,
                                                    inactiveTrackColor = MaterialTheme.colorScheme.outline
                                                )
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }

                    composable("calibrate") {
                        val calibrationVm: CalibrationViewModel = viewModel()
                        DisposableEffect(Unit) {
                            onRegisterCalibrationCallback { frame ->
                                calibrationVm.onFrameCaptured(frame)
                            }
                            onDispose {
                                onRegisterCalibrationCallback(null)
                            }
                        }
                        CalibrationScreen(
                            onDone = { navController.popBackStack() },
                            vm = calibrationVm
                        )
                    }

                    composable("voice_mgmt") {
                        VoiceManagementScreen()
                    }

                    composable("settings") {
                        SideEffect {
                            onOpenSettings()
                            navController.popBackStack()
                        }
                    }

                    composable("switch_cam") {
                        SideEffect {
                            onSwitchCamera()
                            navController.popBackStack()
                        }
                    }

                    composable("clear") {
                        SideEffect {
                            onClearTranscript()
                            navController.popBackStack()
                        }
                    }

                    composable("speak") {
                        SideEffect {
                            onSpeak()
                            navController.popBackStack()
                        }
                    }
                }
            }
        }
    }
    } // LipertyTheme
}