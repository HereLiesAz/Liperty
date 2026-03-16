package com.hereliesaz.liperty.ui

import android.graphics.Bitmap
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.getValue
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
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
    isPaused: Boolean = false,
    isSSIActive: Boolean = false,
    isLipReadActive: Boolean = false,
    currentLensFacing: Int = 0, // 0 for Front, 1 for Back
    vsrSensitivity: Float = 0.5f,
    onVsrSensitivityChange: (Float) -> Unit = {},
    larynxSensitivity: Float = 0.5f,
    onLarynxSensitivityChange: (Float) -> Unit = {},
    isDarkTheme: Boolean = true,
    onRegisterCalibrationCallback: (((Bitmap) -> Unit)?) -> Unit = {}
) {
    val navController = rememberNavController()
    val backgroundColor = if (isDarkTheme) Color(0xFF0A0A0F) else Color(0xFFF5F5F7)
    val textColor = if (isDarkTheme) Color.White else Color.Black

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
                        Box(modifier = Modifier
                            .fillMaxSize()
                            .transformable(state = transformState)
                        ) {
                            // Sensitivity Sliders
                            Column(
                                modifier = Modifier
                                    .align(Alignment.CenterStart)
                                    .padding(16.dp)
                                    .width(48.dp)
                                    .fillMaxHeight(),
                                verticalArrangement = Arrangement.Center,
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                if (isLipReadActive) {
                                    Text("VSR", color = textColor, fontSize = 10.sp)
                                    Slider(
                                        value = vsrSensitivity,
                                        onValueChange = onVsrSensitivityChange,
                                        modifier = Modifier.weight(1f).height(200.dp),
                                        colors = SliderDefaults.colors(thumbColor = textColor, activeTrackColor = textColor)
                                    )
                                }
                                
                                if (isSSIActive) {
                                    Spacer(Modifier.height(16.dp))
                                    Text("LRA", color = Color.Cyan, fontSize = 10.sp)
                                    Slider(
                                        value = larynxSensitivity,
                                        onValueChange = onLarynxSensitivityChange,
                                        modifier = Modifier.weight(1f).height(200.dp),
                                        colors = SliderDefaults.colors(thumbColor = Color.Cyan, activeTrackColor = Color.Cyan)
                                    )
                                }
                            }

                            if (isLipReadActive && transcriptionWords.isNotEmpty()) {
                                FlowRow(
                                    modifier = Modifier
                                        .align(Alignment.Center)
                                        .padding(32.dp),
                                    horizontalArrangement = Arrangement.Center,
                                    verticalArrangement = Arrangement.Center
                                ) {
                                    transcriptionWords.forEachIndexed { index, word ->
                                        val isSelected = index == selectedWordIndex
                                        val confidence = wordConfidences.getOrElse(index) { 0.5f }
                                        // Heatmap: red (0) → amber (0.5) → green (1) at 25 % alpha
                                        val heatColor = lerp(
                                            lerp(Color(0xFFB71C1C), Color(0xFFF57F17), confidence * 2f),
                                            lerp(Color(0xFFF57F17), Color(0xFF1B5E20), (confidence - 0.5f) * 2f),
                                            if (confidence < 0.5f) 0f else 1f
                                        ).copy(alpha = 0.25f)
                                        Text(
                                            text = word,
                                            color = if (isSelected) Color.Cyan else Color.White,
                                            fontSize = fontSize.sp,
                                            fontWeight = if (isSelected) FontWeight.ExtraBold else FontWeight.Normal,
                                            modifier = Modifier
                                                .padding(4.dp)
                                                .clip(RoundedCornerShape(4.dp))
                                                .background(if (isSelected) Color.Cyan.copy(alpha = 0.2f) else heatColor)
                                                .clickable { onWordClick(index) }
                                                .padding(horizontal = 4.dp)
                                        )
                                    }
                                }
                            }

                            if (isRecording) {
                                Text(
                                    text     = if (isPaused) "PAUSED" else if (isSSIActive) "SSI ACTIVE" else "REC",
                                    color    = if (isPaused) Color.Yellow else if (isSSIActive) Color.Cyan else Color.Red,
                                    modifier = Modifier
                                        .align(Alignment.TopEnd)
                                        .padding(16.dp)
                                )
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
}