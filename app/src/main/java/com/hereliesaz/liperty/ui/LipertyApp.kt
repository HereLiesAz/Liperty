package com.HereLiesAz.liperty.ui

import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.camera.view.PreviewView
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
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.HereLiesAz.liperty.voicebox.spike.SpikeScreen
import com.hereliesaz.aznavrail.AzHostActivityLayout
import com.hereliesaz.aznavrail.AzNavHost
import com.hereliesaz.aznavrail.AzTextBox

@Composable
fun LipertyApp(
    previewView: PreviewView,
    overlayView: OverlayView,
    transcriptionText: String,
    onTextChange: (String) -> Unit,
    isRecording: Boolean,
    onSwitchCamera: () -> Unit,
    onOpenSettings: () -> Unit,
    onClearTranscript: () -> Unit,
    onSpeak: () -> Unit
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
                content = Icons.Filled.GraphicEq
            )

            azRailItem(
                id      = "settings",
                text    = "Settings",
                route   = "settings",
                content = Icons.Filled.Settings
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
                AzNavHost(navController = navController, startDestination = "home") {

                    composable("home") {
                        Box(modifier = Modifier.fillMaxSize()) {
                            Box(
                                modifier = Modifier
                                    .align(Alignment.BottomCenter)
                                    .fillMaxWidth()
                                    .padding(16.dp)
                            ) {
                                AzTextBox(
                                    value         = transcriptionText,
                                    onValueChange = onTextChange,
                                    onSubmit      = { onSpeak() },
                                    hint          = "Transcription...",
                                    modifier      = Modifier.fillMaxWidth()
                                )
                            }
                            if (isRecording) {
                                Text(
                                    text     = "REC",
                                    color    = Color.Red,
                                    modifier = Modifier
                                        .align(Alignment.TopEnd)
                                        .padding(16.dp)
                                )
                            }
                        }
                    }

                    // Voice Box spike screen — full Compose, no Activity needed
                    composable("voicebox") {
                        SpikeScreen()
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