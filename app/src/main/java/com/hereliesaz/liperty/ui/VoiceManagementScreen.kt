package com.hereliesaz.liperty.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.FileUpload
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.hereliesaz.liperty.voicebox.VoiceViewModel

@Composable
fun VoiceManagementScreen(
    vm: VoiceViewModel = viewModel()
) {
    val state by vm.uiState.collectAsState()

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { vm.cloneFromUri(it) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0A0A0F))
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        Text(
            "Voice Management",
            color = Color.White,
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold
        )

        Text(
            "Clone your own voice or select a profile for speech synthesis.",
            color = Color(0xFFB0B0CC),
            fontSize = 14.sp
        )

        // Recording Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF12122A)),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Mic,
                    contentDescription = null,
                    tint = if (state.isRecording) Color.Red else Color.Cyan,
                    modifier = Modifier.size(48.dp)
                )
                
                Text(
                    text = if (state.isRecording) "Recording... Speak clearly" else "Ready to Clone",
                    color = Color.White,
                    fontWeight = FontWeight.Medium
                )

                Button(
                    onClick = {
                        if (state.isRecording) vm.stopRecording() else vm.startRecording()
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (state.isRecording) Color(0xFFB71C1C) else Color(0xFF1565C0)
                    )
                ) {
                    Text(if (state.isRecording) "STOP" else "START CLONING")
                }
                
                if (state.isCloning) {
                    CircularProgressIndicator(color = Color.Cyan, modifier = Modifier.size(24.dp))
                }
            }
        }

        // Import WAV Button
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            IconButton(onClick = { filePickerLauncher.launch("audio/*") }) {
                Icon(Icons.Default.FileUpload, contentDescription = "Import WAV", tint = Color.Cyan)
            }
            Text("Import WAV", color = Color.White, fontSize = 16.sp)
        }

        // Voice List
        Text(
            "Saved Voices",
            color = Color.White,
            fontSize = 18.sp,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.align(Alignment.Start)
        )

        LazyColumn(
            modifier = Modifier.fillMaxWidth().weight(1f),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(state.voices) { voice ->
                val isActive = state.activeVoiceName == voice.name
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = if (isActive) Color(0xFF1A1A2E) else Color(0xFF0F0F1A)
                    ),
                    shape = RoundedCornerShape(12.dp),
                    border = if (isActive) CardDefaults.outlinedCardBorder().copy(brush = androidx.compose.ui.graphics.SolidColor(Color.Cyan)) else null
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                            Icon(Icons.Default.RecordVoiceOver, contentDescription = null, tint = if (isActive) Color.Cyan else Color.Gray)
                            Spacer(Modifier.width(12.dp))
                            Text(voice.name, color = Color.White, fontWeight = if (isActive) FontWeight.Bold else FontWeight.Normal)
                        }
                        
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            if (!isActive) {
                                Button(
                                    onClick = { vm.selectVoice(voice.name) },
                                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
                                    modifier = Modifier.height(32.dp)
                                ) {
                                    Text("Select", fontSize = 12.sp)
                                }
                            } else {
                                Text("Active", color = Color.Cyan, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                            }
                            
                            IconButton(onClick = { vm.deleteVoice(voice.name) }, modifier = Modifier.size(32.dp)) {
                                Icon(Icons.Default.Delete, contentDescription = "Delete", tint = Color(0xFF666680))
                            }
                        }
                    }
                }
            }
        }
    }
}
