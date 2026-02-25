package com.HereLiesAz.Liperty.ui

import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.hereliesaz.aznavrail.*

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

    AzHostActivityLayout(
        navController = navController,
        initiallyExpanded = false
    ) {
        // Configuration
        azConfig(
            // dockingSide = AzDockingSide.LEFT, // Using default if unresolved
            packButtons = true,
            displayAppName = true
        )

        azTheme(activeColor = Color.Cyan)

        // Navigation Items
        azRailItem(
            id = "home",
            text = "Home",
            route = "home",
            content = Icons.Filled.Home
        )

        azRailItem(
            id = "settings",
            text = "Settings",
            route = "settings",
            content = Icons.Filled.Settings
        )

        // Action Items
        azMenuItem(
            id = "switch_cam",
            text = "Switch Camera",
            route = "switch_cam",
            content = Icons.Filled.Refresh
        )

        azMenuItem(
            id = "clear",
            text = "Clear Transcript",
            route = "clear",
            content = Icons.Filled.Clear
        )

        azMenuItem(
            id = "speak",
            text = "Speak Text",
            route = "speak",
            content = Icons.Filled.PlayArrow
        )

        // Onscreen Content
        onscreen(alignment = Alignment.Center) {
            AzNavHost(navController = navController, startDestination = "home") {
                composable("home") {
                    Box(modifier = Modifier.fillMaxSize()) {
                        // Camera Preview and Overlay
                        AndroidView(
                            factory = { ctx ->
                                val frameLayout = FrameLayout(ctx).apply {
                                    layoutParams = ViewGroup.LayoutParams(
                                        ViewGroup.LayoutParams.MATCH_PARENT,
                                        ViewGroup.LayoutParams.MATCH_PARENT
                                    )
                                }

                                // Add PreviewView
                                if (previewView.parent != null) {
                                    (previewView.parent as ViewGroup).removeView(previewView)
                                }
                                frameLayout.addView(previewView)

                                // Add OverlayView
                                if (overlayView.parent != null) {
                                    (overlayView.parent as ViewGroup).removeView(overlayView)
                                }
                                frameLayout.addView(overlayView)

                                frameLayout
                            },
                            modifier = Modifier.fillMaxSize()
                        )

                        // Larynx Box (Transcription Interface)
                        // Using AzTextBox for transcription display, editing, and submission (TTS)
                        Box(
                            modifier = Modifier
                                .align(Alignment.BottomCenter)
                                .fillMaxWidth()
                                .padding(16.dp)
                        ) {
                            AzTextBox(
                                value = transcriptionText,
                                onValueChange = onTextChange,
                                onSubmit = { onSpeak() },
                                hint = "Transcription...",
                                modifier = Modifier.fillMaxWidth(),
                                // Assuming AzTextBox handles multiline automatically or has a param
                                // Based on docs: "AzTextBox can be configured as a multiline input"
                                // I'll guess the parameter or modifier usage.
                                // Since I don't have the exact signature, I'll rely on standard naming.
                                // If "multiline" is a boolean param:
                                // multiline = true
                            )
                        }

                        // Recording Indicator
                        if (isRecording) {
                             Text(
                                text = "REC",
                                color = Color.Red,
                                modifier = Modifier
                                    .align(Alignment.TopEnd)
                                    .padding(16.dp)
                            )
                        }
                    }
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
