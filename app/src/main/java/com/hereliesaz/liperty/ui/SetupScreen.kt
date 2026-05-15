package com.hereliesaz.liperty.ui

import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.SkipNext
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.hereliesaz.liperty.setup.ModelDownloadManager

@Composable
fun SetupScreen(
    downloadState: ModelDownloadManager.DownloadState,
    onStartDownload: () -> Unit,
    onRetryModel: (String) -> Unit,
    onSkipOptional: () -> Unit,
    onContinue: () -> Unit
) {
    val hasStarted = downloadState.modelStates.any {
        it.value.status != ModelDownloadManager.Status.PENDING
    }
    val allRequiredDone = downloadState.modelStates.values
        .filter { it.required }
        .all { it.status == ModelDownloadManager.Status.COMPLETE }
    val anyDownloading = downloadState.modelStates.values
        .any { it.status == ModelDownloadManager.Status.DOWNLOADING }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(Modifier.height(48.dp))

            Text(
                text = "Liperty Setup",
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )

            Spacer(Modifier.height(8.dp))

            Text(
                text = "Downloading AI models for on-device processing. " +
                    "After setup, the app works completely offline.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 16.dp)
            )

            Spacer(Modifier.height(24.dp))

            // Overall progress
            if (hasStarted) {
                val totalBytes = downloadState.modelStates.values
                    .filter { it.status != ModelDownloadManager.Status.SKIPPED }
                    .sumOf { it.totalBytes.coerceAtLeast(0) }
                val downloadedBytes = downloadState.modelStates.values
                    .filter { it.status != ModelDownloadManager.Status.SKIPPED }
                    .sumOf { it.progressBytes }
                val overallProgress = if (totalBytes > 0) {
                    downloadedBytes.toFloat() / totalBytes
                } else 0f

                if (anyDownloading) {
                    LinearProgressIndicator(
                        progress = { overallProgress },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = formatBytes(downloadedBytes) + " / " + formatBytes(totalBytes),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Spacer(Modifier.height(16.dp))

            // Model list
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .animateContentSize(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Required models first
                downloadState.modelStates.values
                    .sortedByDescending { it.required }
                    .forEach { model ->
                        ModelDownloadCard(
                            model = model,
                            onRetry = { onRetryModel(model.fileName) }
                        )
                    }
            }

            Spacer(Modifier.height(24.dp))

            // Action buttons
            if (!hasStarted) {
                Button(
                    onClick = onStartDownload,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(Icons.Default.CloudDownload, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Download Models")
                }

                Spacer(Modifier.height(8.dp))

                OutlinedButton(
                    onClick = onSkipOptional,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(Icons.Default.SkipNext, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Download Required Only")
                }
            } else if (allRequiredDone && !anyDownloading) {
                Button(
                    onClick = onContinue,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Continue to App")
                }
            } else if (downloadState.error != null) {
                Text(
                    text = downloadState.error,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(8.dp)
                )
                Button(
                    onClick = onStartDownload,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Retry")
                }
            }

            Spacer(Modifier.height(48.dp))
        }
    }
}

@Composable
private fun ModelDownloadCard(
    model: ModelDownloadManager.ModelDownloadState,
    onRetry: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = when (model.status) {
                ModelDownloadManager.Status.COMPLETE ->
                    MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
                ModelDownloadManager.Status.ERROR ->
                    MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.3f)
                else -> MaterialTheme.colorScheme.surfaceContainerLow
            }
        )
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = model.description,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium
                    )
                    if (model.required) {
                        Text(
                            text = "Required",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }

                when (model.status) {
                    ModelDownloadManager.Status.COMPLETE ->
                        Icon(
                            Icons.Default.CheckCircle,
                            contentDescription = "Complete",
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(24.dp)
                        )
                    ModelDownloadManager.Status.ERROR -> {
                        TextButton(onClick = onRetry) {
                            Icon(
                                Icons.Default.Error,
                                contentDescription = "Error",
                                tint = MaterialTheme.colorScheme.error,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(Modifier.width(4.dp))
                            Text("Retry", color = MaterialTheme.colorScheme.error)
                        }
                    }
                    ModelDownloadManager.Status.SKIPPED ->
                        Text(
                            "Skipped",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.outline
                        )
                    else -> {}
                }
            }

            if (model.status == ModelDownloadManager.Status.DOWNLOADING) {
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress = { model.progressFraction },
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = formatBytes(model.progressBytes) +
                        if (model.totalBytes > 0) " / " + formatBytes(model.totalBytes) else "",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            if (model.status == ModelDownloadManager.Status.ERROR && model.error != null) {
                Spacer(Modifier.height(4.dp))
                Text(
                    text = model.error,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.error
                )
            }
        }
    }
}

private fun formatBytes(bytes: Long): String = when {
    bytes < 1024 -> "$bytes B"
    bytes < 1024 * 1024 -> "${bytes / 1024} KB"
    bytes < 1024 * 1024 * 1024 -> "${"%.1f".format(bytes / (1024.0 * 1024))} MB"
    else -> "${"%.2f".format(bytes / (1024.0 * 1024 * 1024))} GB"
}
