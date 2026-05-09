package com.hereliesaz.liperty.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.ui.res.stringResource
import com.hereliesaz.liperty.R
import com.hereliesaz.liperty.ml.CalibrationViewModel

@Composable
fun CalibrationScreen(
    onDone: () -> Unit,
    vm: CalibrationViewModel = viewModel()
) {
    val state by vm.uiState.collectAsState()

    if (state.calibrationDone) {
        SideEffect {
            onDone()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0A0A0F))
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        // Progress
        LinearProgressIndicator(
            progress = { state.progress },
            modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)),
            color = Color.Cyan,
            trackColor = Color(0xFF1A1A2E)
        )

        Text(
            stringResource(R.string.calibration_title),
            color = Color.White,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold
        )

        Text(
            stringResource(R.string.calibration_instruction),
            color = Color(0xFFB0B0CC),
            fontSize = 14.sp,
            textAlign = TextAlign.Center
        )

        // Phrase Card
        Card(
            modifier = Modifier.fillMaxWidth().weight(1f),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF12122A)),
            shape = RoundedCornerShape(16.dp)
        ) {
            Box(contentAlignment = Alignment.Center, modifier = Modifier.fillMaxSize().padding(24.dp)) {
                Text(
                    text = state.currentPhrase,
                    color = Color.Cyan,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Medium,
                    textAlign = TextAlign.Center,
                    lineHeight = 32.sp
                )
            }
        }

        // Status
        Text(
            text = state.statusMessage,
            color = if (state.isRecording) Color.Red else Color(0xFF9090AA),
            fontSize = 14.sp
        )

        // Controls
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (state.isTraining) {
                CircularProgressIndicator(color = Color.Cyan)
            } else {
                Button(
                    onClick = {
                        if (state.isRecording) vm.stopRecordingAndTrain() else vm.startRecording()
                    },
                    modifier = Modifier.size(120.dp).clip(CircleShape),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (state.isRecording) Color(0xFFB71C1C) else Color(0xFF1565C0)
                    )
                ) {
                    Text(
                        if (state.isRecording) stringResource(R.string.common_stop) else stringResource(R.string.calibration_start),
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
        
        Spacer(Modifier.height(16.dp))
    }
}
