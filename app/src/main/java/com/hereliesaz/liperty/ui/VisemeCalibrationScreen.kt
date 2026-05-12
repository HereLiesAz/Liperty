package com.hereliesaz.liperty.ui

import android.graphics.Bitmap
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.hereliesaz.liperty.personalization.VisemeCalibrationViewModel

/**
 * Compose screen that drives the [VisemeCalibrationViewModel] state
 * machine through a guided viseme-calibration recording session.
 *
 * Camera integration: this screen does NOT host CameraX. It expects
 * the parent (currently `MainActivity`) to be running its normal
 * camera + landmarker + mouth-crop pipeline, and to route the
 * processed 88×88 mouth-crop frames here via the existing
 * `calibrationCallback` mechanism (the same hook the legacy
 * `CalibrationScreen` uses). When the screen mounts, it registers a
 * callback that pushes frames into [VisemeCalibrationViewModel.onFrame];
 * on unmount it unregisters. The ViewModel itself drops frames when
 * it's not in `Phase.Recording`, so the callback is always-on and
 * the per-word recording is gated entirely by user interaction.
 *
 * Design choices:
 * - **Push-to-talk**, not toggle. Holding the button is a natural
 *   metaphor for "speak now" and makes accidental no-op recordings
 *   (user forgot they hit start) impossible.
 * - The current word is the dominant visual element — large, centered,
 *   high contrast. The user should be able to read it from arm's
 *   length while looking at the camera.
 * - We don't show the camera preview here. Showing the user their
 *   own face while they're trying to articulate words encourages
 *   self-correction, which produces non-natural recordings. The
 *   model needs to learn the user's natural articulation, not
 *   their "trying to look correct" articulation.
 */
@Composable
fun VisemeCalibrationScreen(
    vm: VisemeCalibrationViewModel,
    onRegisterFrameCallback: (((Bitmap) -> Unit)?) -> Unit,
    onClose: () -> Unit,
) {
    val state by vm.state.collectAsState()

    // Wire the frame callback for the lifetime of this composable. The
    // VM ignores frames when it's not in Phase.Recording, so this is
    // always-on and per-word gating happens via start/finishRecording.
    DisposableEffect(vm) {
        onRegisterFrameCallback { bmp -> vm.onFrame(bmp) }
        onDispose { onRegisterFrameCallback(null) }
    }

    // NO opaque background on the root — the camera preview + lip-box
    // overlay are rendered by MainActivity as the base layer below this
    // route. The user MUST be able to see themselves and the wireframe
    // crop while reading the prompt, otherwise they have no idea if
    // they're framed correctly. The inner cards get a translucent
    // scrim so text stays legible.
    //
    // Layout: header pinned to the top (small, doesn't cover the face),
    // a vertical Spacer that fills the middle (transparent, the user's
    // face and the wireframe live here), then the prompt card + record
    // button at the bottom. This keeps the face crop visible the whole
    // time so the user can adjust framing if the wireframe is off.
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 24.dp, vertical = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Header(state, onClose)

        // The "see your face" gap. Fills whatever space is between the
        // pinned header at top and the prompt+controls at bottom.
        Spacer(modifier = Modifier.weight(1f))

        when (val phase = state.phase) {
            is VisemeCalibrationViewModel.Phase.Complete -> CompleteCard(
                savedCount = state.savedIds.size,
                onClose = onClose,
                onRestart = { vm.restart() },
            )
            else -> PromptCard(state)
        }

        when (state.phase) {
            is VisemeCalibrationViewModel.Phase.Idle,
            is VisemeCalibrationViewModel.Phase.Recording -> HoldToRecordControls(
                isRecording = state.phase is VisemeCalibrationViewModel.Phase.Recording,
                frameCount = state.recordingFrameCount,
                onStart = { vm.startRecording() },
                onFinish = { vm.finishRecording() },
                onSkip = { vm.skipCurrent() },
            )
            is VisemeCalibrationViewModel.Phase.Saving -> ControlsSaving()
            is VisemeCalibrationViewModel.Phase.Complete -> Unit
        }
    }
}

@Composable
private fun Header(
    state: VisemeCalibrationViewModel.UiState,
    onClose: () -> Unit,
) {
    // Scrim so header text stays legible against the camera preview
    // showing through behind it. Semi-transparent so the user can
    // still see their face if the scrim happens to cover it.
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xCC0A0A0F)),
        shape = RoundedCornerShape(12.dp),
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    "Personalize lipreading",
                    color = Color.White,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                )
                TextButton(onClick = onClose) {
                    Text("Done for now", color = Color(0xFFB0B0CC), fontSize = 14.sp)
                }
            }

            val progress = if (state.totalWords == 0) 0f
                           else state.completedWords.toFloat() / state.totalWords
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp)
                    .clip(RoundedCornerShape(3.dp)),
                color = Color.Cyan,
                trackColor = Color(0xFF1A1A2E),
            )
            Text(
                "${state.completedWords} / ${state.totalWords} words — keep your face in frame",
                color = Color(0xFFB0B0CC),
                fontSize = 12.sp,
            )
        }
    }
}

@Composable
private fun PromptCard(state: VisemeCalibrationViewModel.UiState) {
    // Compact card pinned to the bottom-ish so the user's face stays
    // in view above it. Translucent scrim — text is legible without
    // fully hiding what's behind.
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xE0121224)),
        shape = RoundedCornerShape(16.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                state.currentWord,
                color = Color.Cyan,
                fontSize = 48.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.SansSerif,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.height(6.dp))
            Text(
                "Hold the button, say the word naturally.",
                color = Color(0xFFB0B0CC),
                fontSize = 12.sp,
                textAlign = TextAlign.Center,
            )
            Text(
                "Viseme group: ${state.currentVisemeSeq}",
                color = Color(0xFF6A6A8C),
                fontSize = 9.sp,
            )
        }
    }
}

@Composable
private fun HoldToRecordControls(
    isRecording: Boolean,
    frameCount: Int,
    onStart: () -> Unit,
    onFinish: () -> Unit,
    onSkip: () -> Unit,
) {
    // A single press-and-hold gesture drives the whole record/save
    // lifecycle: onPress -> startRecording (VM transitions to
    // Recording); awaitRelease blocks until the finger lifts; then
    // finishRecording (VM packages frames and transitions to Saving,
    // then back to Idle on the next word). Keeping the gesture on one
    // composable instead of switching between Idle/Recording variants
    // means the user's finger doesn't have to track a moving target
    // mid-recording.
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        TextButton(onClick = onSkip, enabled = !isRecording) {
            Text(
                "Skip",
                color = if (isRecording) Color(0xFF555566) else Color(0xFFB0B0CC),
            )
        }
        Spacer(Modifier.width(8.dp))
        Box(
            modifier = Modifier
                .size(140.dp)
                .clip(CircleShape)
                .background(if (isRecording) Color(0xFFD32F2F) else Color(0xFFE53935))
                .pointerInput(Unit) {
                    detectTapGestures(
                        onPress = {
                            onStart()
                            // awaitRelease suspends until the user
                            // lifts their finger or the gesture is
                            // cancelled (finger drags out, parent
                            // unmounts). In either case we call
                            // finish, which the VM gates on
                            // min-frames so a fat-finger tap won't
                            // produce a save.
                            try { awaitRelease() } finally { onFinish() }
                        }
                    )
                },
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    if (isRecording) "REC" else "HOLD",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = 20.sp,
                )
                if (isRecording) {
                    Text(
                        "$frameCount fr",
                        color = Color.White.copy(alpha = 0.85f),
                        fontSize = 11.sp,
                    )
                } else {
                    Text(
                        "and speak",
                        color = Color.White.copy(alpha = 0.85f),
                        fontSize = 11.sp,
                    )
                }
            }
        }
    }
}

@Composable
private fun ControlsSaving() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        CircularProgressIndicator(color = Color.Cyan, modifier = Modifier.size(36.dp))
        Spacer(Modifier.width(12.dp))
        Text("Saving recording...", color = Color.White, fontSize = 14.sp)
    }
}

@Composable
private fun CompleteCard(savedCount: Int, onClose: () -> Unit, onRestart: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(220.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF12122A)),
        shape = RoundedCornerShape(16.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("All done.", color = Color.Cyan, fontSize = 28.sp, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(8.dp))
            Text(
                "$savedCount recordings saved. They'll be used to train a personalized lipreading model on this device.",
                color = Color(0xFFB0B0CC),
                fontSize = 13.sp,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.height(20.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                TextButton(onClick = onRestart) {
                    Text("Record more", color = MaterialTheme.colorScheme.primary)
                }
                Button(onClick = onClose) { Text("Close") }
            }
        }
    }
}
