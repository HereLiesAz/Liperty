package com.hereliesaz.liperty.voicebox

import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

@RunWith(RobolectricTestRunner::class)
class AudioRouterTest {

    private lateinit var router: AudioRouter
    private lateinit var context: Context
    private lateinit var audioManager: AudioManager

    @Before
    fun setUp() {
        context = RuntimeEnvironment.getApplication()
        audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        router = AudioRouter(context)
    }

    // ── Input routing ────────────────────────────────────────────────────

    @Test
    fun `getPhoneMic returns builtin mic or null`() {
        val mic = router.getPhoneMic()
        if (mic != null) {
            assertEquals(
                "Phone mic should be TYPE_BUILTIN_MIC",
                AudioDeviceInfo.TYPE_BUILTIN_MIC,
                mic.type
            )
        }
        // null is also acceptable in emulated environments without a builtin mic
    }

    @Test
    fun `getOptimalInputDevice does not crash`() {
        val device = router.getOptimalInputDevice()
        // Should return an AudioDeviceInfo or null without throwing
        if (device != null) {
            assertTrue("Device type should be non-negative", device.type >= 0)
        }
    }

    // ── Output routing ───────────────────────────────────────────────────

    @Test
    fun `getOptimalOutputDevice does not crash`() {
        val device = router.getOptimalOutputDevice()
        // Should return an AudioDeviceInfo or null without throwing
        if (device != null) {
            assertTrue("Device type should be non-negative", device.type >= 0)
        }
    }

    // ── Bone conduction detection ────────────────────────────────────────

    @Test
    fun `isUsingBoneConduction returns false without BT`() {
        // Robolectric emulates a basic device without Bluetooth peripherals,
        // so bone conduction should never be detected.
        assertFalse(
            "Bone conduction should be false in emulated environment without BT",
            router.isUsingBoneConduction()
        )
    }

    // ── Full-duplex configuration ────────────────────────────────────────

    @Test
    fun `configureFullDuplex sets communication mode`() {
        router.configureFullDuplex()

        assertEquals(
            "Audio mode should be MODE_IN_COMMUNICATION after configureFullDuplex",
            AudioManager.MODE_IN_COMMUNICATION,
            audioManager.mode
        )
    }

    @Test
    fun `releaseFullDuplex restores previous mode`() {
        // Set a known starting mode
        audioManager.mode = AudioManager.MODE_NORMAL

        router.configureFullDuplex()
        assertEquals(
            "Should be in communication mode after configure",
            AudioManager.MODE_IN_COMMUNICATION,
            audioManager.mode
        )

        router.releaseFullDuplex()
        assertEquals(
            "Should restore MODE_NORMAL after release",
            AudioManager.MODE_NORMAL,
            audioManager.mode
        )
    }

    @Test
    fun `releaseFullDuplex without configure is safe`() {
        // Set a known mode before calling release without prior configure
        audioManager.mode = AudioManager.MODE_NORMAL

        // Should not throw any exception
        router.releaseFullDuplex()

        // Mode should remain unchanged since previousMode was null
        assertEquals(
            "Mode should remain unchanged after orphaned releaseFullDuplex",
            AudioManager.MODE_NORMAL,
            audioManager.mode
        )
    }
}
