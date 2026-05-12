package com.hereliesaz.liperty.ui

import android.graphics.Bitmap
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.camera.view.PreviewView
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
import androidx.compose.runtime.LaunchedEffect
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.hereliesaz.liperty.R
import com.hereliesaz.liperty.ml.CalibrationViewModel
import com.hereliesaz.aznavrail.AzHostActivityLayout
import com.hereliesaz.aznavrail.AzNavHost

@Composable
fun LipertyApp(
    previewView: PreviewView,
    overlayView: OverlayView,
    transcriptionWords: List<String>,
    wordConfidences: List<Float> = emptyList(),
    /** Live-preview words from the latest inference window that aren't yet
     *  committed. Rendered at reduced alpha + italic as the dimmer tail of
     *  the transcript so users see speech as it's mouthed before the next
     *  window confirms it. */
    liveWords: List<String> = emptyList(),
    liveWordConfidences: List<Float> = emptyList(),
    selectedWordIndex: Int,
    onWordClick: (Int) -> Unit,
    isRecording: Boolean,
    onSwitchCamera: () -> Unit,
    onOpenSettings: () -> Unit,
    onClearTranscript: () -> Unit,
    onSpeak: () -> Unit,
    onToggleBC: () -> Unit = {},
    onToggleLipRead: () -> Unit = {},
    onToggleEL: () -> Unit = {},
    isPaused: Boolean = false,
    isBCActive: Boolean = false,
    isLipReadActive: Boolean = false,
    isELActive: Boolean = false,
    currentLensFacing: Int = 0, // 0 for Front, 1 for Back
    vsrSensitivity: Float = 0.5f,
    onVsrSensitivityChange: (Float) -> Unit = {},
    larynxSensitivity: Float = 0.5f,
    onLarynxSensitivityChange: (Float) -> Unit = {},
    carrierF0: Float = 120f,
    onCarrierF0Change: (Float) -> Unit = {},
    isDarkTheme: Boolean = true,
    onRegisterCalibrationCallback: (((Bitmap) -> Unit)?) -> Unit = {}
) {
    val navController = rememberNavController()
    val context = LocalContext.current

    LipertyTheme(isDarkTheme = isDarkTheme) {
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
                text    = context.getString(R.string.nav_home),
                route   = "home",
                content = Icons.Filled.Home,
                color   = Color.White
            )

            azRailToggle(
                id            = "lipread",
                isChecked     = isLipReadActive,
                toggleOnText  = context.getString(R.string.mode_lipread_on),
                toggleOffText = context.getString(R.string.mode_lipread_off),
                onClick       = { onToggleLipRead() },
                color         = Color.White
            )

            azRailToggle(
                id            = "voicebox",
                isChecked     = isBCActive,
                toggleOnText  = context.getString(R.string.mode_bc_on),
                toggleOffText = context.getString(R.string.mode_bc_off),
                onClick       = { onToggleBC() },
                color         = Color.White
            )
            
            azRailToggle(
                id            = "el_translator",
                isChecked     = isELActive,
                toggleOnText  = context.getString(R.string.mode_el_on),
                toggleOffText = context.getString(R.string.mode_el_off),
                onClick       = { onToggleEL() },
                color         = Color.Magenta
            )

            azRailToggle(
                id            = "switch_cam",
                isChecked     = currentLensFacing == 1, // 1 for Back
                toggleOnText  = context.getString(R.string.mode_camera_back),
                toggleOffText = context.getString(R.string.mode_camera_front),
                onClick       = { onSwitchCamera() },
                color         = Color.White
            )

            azRailItem(
                id      = "voice_mgmt",
                text    = context.getString(R.string.nav_voice),
                route   = "voice_mgmt",
                content = Icons.Filled.RecordVoiceOver,
                color   = Color.White
            )

            azRailItem(
                id      = "calibrate",
                text    = context.getString(R.string.nav_tweak),
                route   = "calibrate",
                content = Icons.Filled.Refresh,
                color   = Color.White
            )

            azRailItem(
                id      = "settings",
                text    = context.getString(R.string.nav_settings),
                route   = "settings",
                content = Icons.Filled.Settings,
                color   = Color.White
            )

            // ── Action menu items ─────────────────────────────────────────
            azMenuItem(
                id      = "clear",
                text    = context.getString(R.string.nav_clear_transcript),
                route   = "clear",
                content = Icons.Filled.Clear
            )

            azMenuItem(
                id      = "speak",
                text    = context.getString(R.string.nav_speak_text),
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
                        val anyModeActive = isLipReadActive || isBCActive || isELActive
                        val activeColor = when {
                            isELActive -> MaterialTheme.colorScheme.tertiary
                            isBCActive -> MaterialTheme.colorScheme.secondary
                            else       -> MaterialTheme.colorScheme.primary
                        }

                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .transformable(state = transformState)
                        ) {
                            // ── Top: animated scanning line ────────────────
                            if (anyModeActive) {
                                LinearProgressIndicator(
                                    modifier   = Modifier.fillMaxWidth(),
                                    color      = activeColor,
                                    trackColor = activeColor.copy(alpha = 0.15f)
                                )
                            }

                            // ── Middle: transcription overlaid on camera ───
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .weight(1f)
                            ) {
                                // Words — committed transcript + live preview tail.
                                // Live words are the latest inference window's new tail
                                // that hasn't been confirmed yet by the next window;
                                // they appear dimmer + italic and aren't clickable.
                                if (transcriptionWords.isNotEmpty() || liveWords.isNotEmpty()) {
                                    FlowRow(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .align(Alignment.Center)
                                            .background(Color.Black.copy(alpha = 0.45f))
                                            .padding(horizontal = 24.dp, vertical = 16.dp),
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
                                                    activeColor
                                                else
                                                    Color.White,
                                                fontSize = fontSize.sp,
                                                fontWeight = if (isSelected) FontWeight.ExtraBold else FontWeight.Normal,
                                                modifier = Modifier
                                                    .padding(4.dp)
                                                    .clip(RoundedCornerShape(8.dp))
                                                    .background(
                                                        if (isSelected)
                                                            activeColor.copy(alpha = 0.28f)
                                                        else
                                                            heatColor
                                                    )
                                                    .clickable { onWordClick(index) }
                                                    .padding(horizontal = 6.dp, vertical = 2.dp)
                                            )
                                        }
                                        liveWords.forEach { liveWord ->
                                            Text(
                                                text = liveWord,
                                                color = Color.White.copy(alpha = 0.55f),
                                                fontSize = fontSize.sp,
                                                fontWeight = FontWeight.Light,
                                                fontStyle = androidx.compose.ui.text.font.FontStyle.Italic,
                                                modifier = Modifier
                                                    .padding(4.dp)
                                                    .padding(horizontal = 6.dp, vertical = 2.dp)
                                            )
                                        }
                                    }
                                } else if (!anyModeActive) {
                                    // Nothing active — show subtle hint in center
                                    Text(
                                        text = stringResource(R.string.hint_enable_mode),
                                        color = MaterialTheme.colorScheme.outlineVariant,
                                        style = MaterialTheme.typography.bodyMedium,
                                        textAlign = TextAlign.Center,
                                        modifier = Modifier.align(Alignment.Center)
                                    )
                                }
                                // Mode active but no words yet — silent, line at top indicates state

                                // Status badge — top-right corner
                                if (isRecording) {
                                    val (badgeText, badgeColor) = when {
                                        isPaused        -> stringResource(R.string.badge_paused)  to Color(0xFFFFD600)
                                        isELActive      -> stringResource(R.string.badge_el)      to MaterialTheme.colorScheme.tertiary
                                        isBCActive      -> stringResource(R.string.badge_bc)      to MaterialTheme.colorScheme.primary
                                        isLipReadActive -> stringResource(R.string.badge_reading) to Color(0xFFFF5252)
                                        else            -> stringResource(R.string.badge_rec)     to Color(0xFFFF5252)
                                    }
                                    Box(
                                        modifier = Modifier
                                            .align(Alignment.TopEnd)
                                            .padding(12.dp)
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

                            // ── Bottom: sensitivity slider(s) ──────────────
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
                                                    text = stringResource(R.string.slider_detection_sensitivity),
                                                    style = MaterialTheme.typography.labelLarge,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                                Text(
                                                    text = stringResource(R.string.format_percent, (vsrSensitivity * 100).toInt()),
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
                                                    thumbColor         = MaterialTheme.colorScheme.primary,
                                                    activeTrackColor   = MaterialTheme.colorScheme.primary,
                                                    inactiveTrackColor = MaterialTheme.colorScheme.outline
                                                )
                                            )
                                        }

                                        if (isBCActive || isELActive) {
                                            if (isLipReadActive) Spacer(Modifier.height(4.dp))
                                            val sliderColor = if (isELActive)
                                                MaterialTheme.colorScheme.tertiary
                                            else
                                                MaterialTheme.colorScheme.secondary
                                            val sliderLabel = if (isELActive)
                                                stringResource(R.string.slider_el_sensitivity)
                                            else
                                                stringResource(R.string.slider_bc_sensitivity)
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
                                                    text = stringResource(R.string.format_percent, (larynxSensitivity * 100).toInt()),
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

                                            // F0 carrier pitch slider — only visible in BC mode
                                            if (isBCActive) {
                                                Spacer(Modifier.height(4.dp))
                                                Row(
                                                    modifier = Modifier.fillMaxWidth(),
                                                    horizontalArrangement = Arrangement.SpaceBetween,
                                                    verticalAlignment = Alignment.CenterVertically
                                                ) {
                                                    Text(
                                                        text = stringResource(R.string.slider_carrier_pitch),
                                                        style = MaterialTheme.typography.labelLarge,
                                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                                    )
                                                    Text(
                                                        text = stringResource(R.string.format_hz, carrierF0.toInt()),
                                                        style = MaterialTheme.typography.labelLarge,
                                                        fontWeight = FontWeight.Bold,
                                                        color = MaterialTheme.colorScheme.secondary
                                                    )
                                                }
                                                Slider(
                                                    value = carrierF0,
                                                    onValueChange = onCarrierF0Change,
                                                    valueRange = 80f..200f,
                                                    modifier = Modifier.fillMaxWidth(),
                                                    colors = SliderDefaults.colors(
                                                        thumbColor         = MaterialTheme.colorScheme.secondary,
                                                        activeTrackColor   = MaterialTheme.colorScheme.secondary,
                                                        inactiveTrackColor = MaterialTheme.colorScheme.outline
                                                    )
                                                )
                                            }
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
                        VoiceManagementScreen(
                            onNavigateToImportWizard = {
                                navController.navigate("voice_import_wizard")
                            }
                        )
                    }

                    composable("voice_import_wizard") {
                        val voiceVm: com.hereliesaz.liperty.voicebox.VoiceViewModel = androidx.lifecycle.viewmodel.compose.viewModel()
                        VoiceImportWizardScreen(
                            vm = voiceVm,
                            onDismiss = { navController.popBackStack() }
                        )
                    }

                    composable("settings") {
                        LaunchedEffect(Unit) {
                            onOpenSettings()
                            navController.popBackStack()
                        }
                    }

                    composable("switch_cam") {
                        LaunchedEffect(Unit) {
                            onSwitchCamera()
                            navController.popBackStack()
                        }
                    }

                    composable("clear") {
                        LaunchedEffect(Unit) {
                            onClearTranscript()
                            navController.popBackStack()
                        }
                    }

                    composable("speak") {
                        LaunchedEffect(Unit) {
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