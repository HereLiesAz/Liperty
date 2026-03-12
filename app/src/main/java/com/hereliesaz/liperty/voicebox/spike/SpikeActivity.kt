package com.hereliesaz.liperty.voicebox.spike

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Phase 0 Spike Activity.
 *
 * Throwaway diagnostic UI — NOT part of the final Liperty nav graph.
 * To launch during development, start this Activity directly:
 *
 *   adb shell am start -n com.hereliesaz.liperty/.voicebox.spike.SpikeActivity
 *
 * Or add a temporary debug menu entry in MainActivity and remove before release.
 *
 * UI provides:
 *   - Throat-placement guidance
 *   - Live accelerometer magnitude indicator (confirms contact vibration)
 *   - Record / Stop button
 *   - Result display with file paths for adb pull
 */
class SpikeActivity : ComponentActivity() {

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        permissionGranted.value = granted
    }

    private val permissionGranted = mutableStateOf(false)

    // Live accelerometer magnitude — updated on sensor thread, read on UI thread.
    // Volatile is sufficient here; we don't need atomic for a float display value.
    @Volatile private var accelMagnitude = 0f

    private val sensorManager by lazy { getSystemService(SensorManager::class.java) }
    private val accelerometer  by lazy { sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) }

    private val accelListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            val x = event.values[0]
            val y = event.values[1]
            val z = event.values[2]
            // Subtract gravity component (≈9.8 m/s²) to get dynamic acceleration only.
            // Gravity dominates the static signal; we want vibration magnitude.
            val dynamic = sqrt(x * x + y * y + z * z) - SensorManager.GRAVITY_EARTH
            accelMagnitude = abs(dynamic)
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        permissionGranted.value = ContextCompat.checkSelfPermission(
            this, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

        setContent { SpikeScreen() }
    }

    override fun onResume() {
        super.onResume()
        sensorManager.registerListener(accelListener, accelerometer, SensorManager.SENSOR_DELAY_FASTEST)
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(accelListener)
    }

    // -------------------------------------------------------------------------
    // Compose UI
    // -------------------------------------------------------------------------

    @Composable
    private fun SpikeScreen() {
        val recorder    = remember { SignalRecorder(this) }
        val scope       = rememberCoroutineScope()
        var isRecording by remember { mutableStateOf(false) }
        var statusText  by remember { mutableStateOf("Press Record to begin") }
        var resultText  by remember { mutableStateOf("") }
        var liveAccel   by remember { mutableStateOf(0f) }

        // Poll accelMagnitude for display — 20 Hz is plenty for a VU-meter.
        LaunchedEffect(Unit) {
            while (true) {
                liveAccel = accelMagnitude
                delay(50L)
            }
        }

        // Animate the vibration indicator bar
        val barWidth by animateFloatAsState(
            targetValue  = (liveAccel * 20f).coerceIn(0f, 1f),
            animationSpec = tween(durationMillis = 80),
            label        = "accel_bar"
        )

        MaterialTheme {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color(0xFF0D0D0D))
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(20.dp, Alignment.CenterVertically)
            ) {

                Text(
                    "Voice Box Spike",
                    color     = Color.White,
                    fontSize  = 26.sp,
                    fontWeight = FontWeight.Bold
                )

                // Placement guidance
                Card(
                    colors  = CardDefaults.cardColors(containerColor = Color(0xFF1A1A2E)),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("📱 Placement", color = Color(0xFF7EB6FF), fontWeight = FontWeight.SemiBold)
                        Spacer(Modifier.height(8.dp))
                        Text(
                            "Hold the phone vertically with the screen facing OUT.\n" +
                                    "Press the back of the device firmly against your throat,\n" +
                                    "just below the Adam's apple.\n\n" +
                                    "Speak or hum normally. The vibration bar below should\n" +
                                    "react to your voice.",
                            color    = Color(0xFFCCCCCC),
                            fontSize = 14.sp,
                            lineHeight = 20.sp
                        )
                    }
                }

                // Live vibration indicator
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        "LARYNGEAL VIBRATION",
                        color    = Color(0xFF888888),
                        fontSize = 11.sp,
                        letterSpacing = 2.sp
                    )
                    Spacer(Modifier.height(8.dp))
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(24.dp)
                            .clip(MaterialTheme.shapes.small)
                            .background(Color(0xFF222222))
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth(barWidth)
                                .fillMaxHeight()
                                .background(
                                    if (barWidth > 0.6f) Color(0xFF00E676)
                                    else if (barWidth > 0.2f) Color(0xFFFFD740)
                                    else Color(0xFF455A64)
                                )
                        )
                    }
                    Spacer(Modifier.height(4.dp))
                    Text(
                        "%.3f m/s²".format(liveAccel),
                        color    = Color(0xFF666666),
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }

                // Status
                Text(
                    statusText,
                    color     = if (isRecording) Color(0xFFFF5252) else Color(0xFFAAAAAA),
                    fontSize  = 14.sp,
                    textAlign = TextAlign.Center
                )

                // Record button
                if (!permissionGranted.value) {
                    Button(
                        onClick = { permissionLauncher.launch(Manifest.permission.RECORD_AUDIO) },
                        modifier = Modifier.size(160.dp).clip(CircleShape)
                    ) {
                        Text("Grant\nMic\nPermission", textAlign = TextAlign.Center)
                    }
                } else {
                    Button(
                        onClick = {
                            if (isRecording) {
                                recorder.stop()
                                isRecording = false
                                statusText  = "Stopped early."
                            } else {
                                isRecording = true
                                resultText  = ""
                                statusText  = "Recording 10 seconds…"
                                scope.launch {
                                    recorder.record(
                                        durationMs = 10_000L,
                                        onComplete = { result ->
                                            isRecording = false
                                            statusText  = "Done."
                                            resultText  = buildString {
                                                appendLine("✅ Recording complete")
                                                appendLine()
                                                appendLine("Audio: ${result.audioFile.name}")
                                                appendLine("  ${result.audioSamples} samples @ 16kHz")
                                                appendLine()
                                                appendLine("Accel: ${result.accelFile.name}")
                                                appendLine("  ${result.accelSamples} samples")
                                                appendLine()
                                                appendLine("Duration: ${result.durationMs} ms")
                                                appendLine()
                                                appendLine("Pull with:")
                                                appendLine("adb pull /sdcard/Android/data/")
                                                appendLine("com.hereliesaz.liperty/files/")
                                            }
                                        },
                                        onError = { msg ->
                                            isRecording = false
                                            statusText  = "Error: $msg"
                                        }
                                    )
                                }
                            }
                        },
                        modifier = Modifier
                            .size(140.dp)
                            .clip(CircleShape),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isRecording) Color(0xFFB71C1C) else Color(0xFF1565C0)
                        )
                    ) {
                        Text(
                            if (isRecording) "STOP" else "RECORD",
                            fontSize   = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                // Result
                if (resultText.isNotEmpty()) {
                    Card(
                        colors   = CardDefaults.cardColors(containerColor = Color(0xFF0A1628)),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            resultText,
                            modifier   = Modifier.padding(12.dp),
                            color      = Color(0xFF80CBC4),
                            fontSize   = 12.sp,
                            fontFamily = FontFamily.Monospace,
                            lineHeight = 18.sp
                        )
                    }
                }
            }
        }
    }
}
