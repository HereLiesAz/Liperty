package com.hereliesaz.liperty.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/** Material 3 Expressive dark colour scheme for Liperty. */
private val LipertyDarkColors = darkColorScheme(
    primary             = Color(0xFF00E5FF), // Bright cyan — the app's active colour
    onPrimary           = Color(0xFF003640),
    primaryContainer    = Color(0xFF004D5C),
    onPrimaryContainer  = Color(0xFF80F8FF),
    secondary           = Color(0xFF7C4DFF), // Expressive violet accent
    onSecondary         = Color(0xFF1A0050),
    secondaryContainer  = Color(0xFF2E0085),
    onSecondaryContainer= Color(0xFFD5AAFF),
    tertiary            = Color(0xFF69FF47), // EL green
    onTertiary          = Color(0xFF003900),
    tertiaryContainer   = Color(0xFF004E00),
    onTertiaryContainer = Color(0xFF8FFF6A),
    error               = Color(0xFFFF5252),
    onError             = Color(0xFF5A0000),
    errorContainer      = Color(0xFF93000A),
    onErrorContainer    = Color(0xFFFFDAD6),
    background          = Color(0xFF0A0A0F),
    onBackground        = Color(0xFFE6E1E5),
    surface             = Color(0xFF12121F),
    onSurface           = Color(0xFFE6E1E5),
    surfaceVariant      = Color(0xFF1A1A2E),
    onSurfaceVariant    = Color(0xFFCAC4D0),
    surfaceContainerLow = Color(0xFF14141F),
    surfaceContainer    = Color(0xFF1C1C2A),
    surfaceContainerHigh= Color(0xFF242435),
    outline             = Color(0xFF3D3D5C),
    outlineVariant      = Color(0xFF25253B),
)

@Composable
fun LipertyTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LipertyDarkColors,
        content     = content
    )
}
