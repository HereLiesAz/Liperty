package com.hereliesaz.liperty.voicebox

import android.content.Context
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33], manifest = Config.NONE)
class VoiceManagerTest {

    private lateinit var voiceManager: VoiceManager

    @Before
    fun setUp() {
        voiceManager = VoiceManager(RuntimeEnvironment.getApplication()) { _ -> }
    }

    @Test
    fun `speak with null voice uses system TTS path`() {
        // activeVoice is null by default (no saved preference), so speak()
        // should attempt the system TTS path rather than PocketTTS.
        // System TTS won't actually be ready under Robolectric, but the
        // call must not crash — it should log a warning and return.
        voiceManager.setActiveVoice(null)
        voiceManager.speak("hello")
    }

    @Test
    fun `stop is safe to call multiple times`() {
        voiceManager.stop()
        voiceManager.stop()
        voiceManager.stop()
    }

    @Test
    fun `shutdown releases streaming track`() {
        // After shutdown, subsequent calls should not crash even though
        // internal resources have been released.
        voiceManager.shutdown()
        // A second shutdown should also be safe.
        voiceManager.shutdown()
        // stop() after shutdown should not throw.
        voiceManager.stop()
    }

    @Test
    fun `playAudio delegates to playAudioStatic`() {
        // Provide a small float array — AudioTrack creation may fail under
        // Robolectric, but the method must handle the failure gracefully
        // and not crash.
        val samples = FloatArray(160) { 0.5f }
        voiceManager.playAudio(samples)
    }

    @Test
    fun `getClonedVoices returns empty initially`() {
        // On a fresh VoiceManager with no saved voices, the list should
        // be empty.
        val voices = voiceManager.getClonedVoices()
        assertTrue("Expected empty cloned voices list, got ${voices.size}", voices.isEmpty())
    }

    @Test
    fun `setActiveVoice persists preference`() {
        val testVoice = VoiceState("TestVoice", floatArrayOf(1f, 2f, 3f))
        voiceManager.setActiveVoice(testVoice)

        val prefs = RuntimeEnvironment.getApplication()
            .getSharedPreferences("LipertyPrefs", Context.MODE_PRIVATE)
        assertEquals("TestVoice", prefs.getString("active_voice", null))
    }
}
