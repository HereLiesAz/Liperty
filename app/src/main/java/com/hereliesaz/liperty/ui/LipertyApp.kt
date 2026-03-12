package com.hereliesaz.liperty.ui

import android.graphics.Bitmap
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.rememberTransformableState
import androidx.compose.foundation.gestures.transformable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.hereliesaz.liperty.ml.CalibrationViewModel
import com.hereliesaz.liperty.voicebox.spike.SpikeScreen
import com.hereliesaz.liperty.ui.CalibrationScreen
import com.hereliesaz.aznavrail.AzHostActivityLayout
import com.hereliesaz.aznavrail.AzNavHost

@Composable
fun LipertyApp(
    previewView: PreviewView,
    overlayView: OverlayView,
    transcriptionText: String,
    isRecording: Boolean,
    onSwitchCamera: () -> Unit,
    onOpenSettings: () -> Unit,
    onClearTranscript: () -> Unit,
    onSpeak: () -> Unit,
    onToggleSSI: () -> Unit = {},
    isPaused: Boolean = false,
    isSSIActive: Boolean = false,
    onRegisterCalibrationCallback: (((Bitmap) -> Unit)?) -> Unit = {}
) {
    val navController = rememberNavController()

    Box(modifier = Modifier.fillMaxSize()) {

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
                content = Icons.Filled.Home
            )

            azRailItem(
                id      = "voicebox",
                text    = "Voice Box",
                route   = "voicebox",
                content = Icons.Filled.GraphicEq,
                active  = isSSIActive
            )

            azRailItem(
                id      = "settings",
                text    = "Settings",
                route   = "settings",
                content = Icons.Filled.Settings
            )

            azRailItem(
                id      = "calibrate",
                text    = "Personalize",
                route   = "calibrate",
                content = Icons.Filled.Refresh // Use appropriate icon if available
            )

            azRailItem(
                id      = "switch_cam",
                text    = "Switch Camera",
                route   = "switch_cam",
                content = Icons.Filled.Refresh
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
                            if (transcriptionText.isNotEmpty()) {
                                Text(
                                    text = transcriptionText,
                                    color = Color.White,
                                    fontSize = fontSize.sp,
                                    fontWeight = FontWeight.Bold,
                                    textAlign = TextAlign.Center,
                                    modifier = Modifier
                                        .align(Alignment.Center)
                                        .padding(32.dp)
                                        .background(Color.Black.copy(alpha = 0.4f))
                                        .padding(8.dp)
                                )
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

                    // Voice Box toggle — starts artificial larynx carrier
                    composable("voicebox") {
                        SideEffect {
                            onToggleSSI()
                            navController.popBackStack()
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