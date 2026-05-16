package com.hereliesaz.liperty.voicebox

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.hereliesaz.liperty.voicebox.cloning.AudioPreprocessor
import com.hereliesaz.liperty.voicebox.cloning.SpeakerClusterer
import com.hereliesaz.liperty.voicebox.cloning.VoiceQualityTier
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Tests for [VoiceViewModel] state machine.
 *
 * Covers synchronous state transitions (selectSpeaker, confirmSpeakerSelection,
 * resetImport, selectVoice, deleteVoice). Async operations (startImportProcessing,
 * saveImportedProfile) require dependency injection of dispatchers and are deferred
 * to integration tests.
 *
 * Uses Robolectric with sdk=34 per the project's Robolectric SDK 37 workaround.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VoiceViewModelTest {

    private lateinit var vm: VoiceViewModel

    @Before
    fun setUp() {
        val app = ApplicationProvider.getApplicationContext<Application>()
        vm = VoiceViewModel(app)
    }

    // ── Initial state ────────────────────────────────────────────────────

    @Test
    fun `initial UI state is default`() {
        val state = vm.uiState.value
        assertFalse(state.isRecording)
        assertFalse(state.isCloning)
        assertFalse(state.isNamingVoice)
        assertEquals("Ready", state.statusMessage)
    }

    @Test
    fun `initial import state is IDLE`() {
        assertEquals(ImportStep.IDLE, vm.importState.value.step)
    }

    // ── Speaker selection ────────────────────────────────────────────────

    @Test
    fun `selectSpeaker sets selectedClusterId`() {
        vm.selectSpeaker(2)
        assertEquals(2, vm.importState.value.selectedClusterId)
    }

    @Test
    fun `selectSpeaker can change selection`() {
        vm.selectSpeaker(1)
        vm.selectSpeaker(3)
        assertEquals(3, vm.importState.value.selectedClusterId)
    }

    // ── Reset ────────────────────────────────────────────────────────────

    @Test
    fun `resetImport returns to IDLE`() {
        // Simulate being in an error state
        vm.selectSpeaker(1) // modify some state
        vm.resetImport()
        val state = vm.importState.value
        assertEquals(ImportStep.IDLE, state.step)
        assertNull(state.selectedClusterId)
        assertTrue(state.segments.isEmpty())
        assertTrue(state.clusters.isEmpty())
    }

    // ── Voice selection and deletion ─────────────────────────────────────

    @Test
    fun `selectVoice updates active voice name in UI state`() {
        vm.selectVoice("test-voice")
        assertEquals("test-voice", vm.uiState.value.activeVoiceName)
    }

    @Test
    fun `cancelCloning resets naming flag`() {
        vm.cancelCloning()
        assertFalse(vm.uiState.value.isNamingVoice)
    }

    // ── Data classes ─────────────────────────────────────────────────────

    @Test
    fun `ImportState defaults are sensible`() {
        val state = ImportState()
        assertEquals(ImportStep.IDLE, state.step)
        assertEquals(0f, state.progressPercent, 0.001f)
        assertEquals("", state.progressMessage)
        assertTrue(state.segments.isEmpty())
        assertTrue(state.clusters.isEmpty())
        assertNull(state.selectedClusterId)
        assertEquals(0f, state.qualityScore, 0.001f)
        assertEquals(0L, state.totalSpeechDurationMs)
        assertEquals(0, state.sampleCount)
        assertNull(state.error)
    }

    @Test
    fun `VoiceUiState defaults are sensible`() {
        val state = VoiceUiState()
        assertTrue(state.voices.isEmpty())
        assertTrue(state.profiles.isEmpty())
        assertNull(state.activeVoiceName)
        assertFalse(state.isRecording)
        assertFalse(state.isCloning)
        assertFalse(state.isNamingVoice)
    }

    // ── Coached recording state ──────────────────────────────────────────

    @Test
    fun `initial coach state is empty`() {
        val state = vm.coachState.value
        assertEquals(0, state.clipCount)
        assertEquals(0, state.totalSpeechSeconds)
        assertEquals(VoiceQualityTier.NONE, state.qualityTier)
        assertFalse(state.isRecording)
        assertFalse(state.isSaving)
        assertFalse(state.isSaved)
        assertNull(state.savedProfileName)
        assertNull(state.error)
    }

    @Test
    fun `startCoachedSession resets state`() {
        // Simulate prior state by saving + cancel pattern.
        vm.startCoachedSession()
        vm.cancelCoachedSession()
        vm.startCoachedSession()
        val state = vm.coachState.value
        assertEquals(0, state.clipCount)
        assertEquals(VoiceQualityTier.NONE, state.qualityTier)
        assertFalse(state.isRecording)
        assertFalse(state.isSaved)
    }

    @Test
    fun `cancelCoachedSession returns to empty state`() {
        vm.startCoachedSession()
        vm.cancelCoachedSession()
        val state = vm.coachState.value
        assertEquals(0, state.clipCount)
        assertEquals(VoiceQualityTier.NONE, state.qualityTier)
    }

    @Test
    fun `discardLastCoachedClip is safe when no clips captured`() {
        // No-op — must not throw, must not advance state.
        vm.startCoachedSession()
        vm.discardLastCoachedClip()
        assertEquals(0, vm.coachState.value.clipCount)
    }

    @Test
    fun `finishCoachedSession with no clips surfaces a status message`() {
        vm.startCoachedSession()
        vm.finishCoachedSession("EmptyVoice")
        val state = vm.coachState.value
        // Should not enter the saving state, should not mark as saved.
        assertFalse(state.isSaving)
        assertFalse(state.isSaved)
        assertEquals(0, state.clipCount)
        assertTrue(state.statusMessage.contains("No clips", ignoreCase = true))
    }

    @Test
    fun `CoachState defaults are sensible`() {
        val state = CoachState()
        assertEquals(0, state.clipCount)
        assertEquals(0, state.totalSpeechSeconds)
        assertEquals(VoiceQualityTier.NONE, state.qualityTier)
        assertEquals(0, state.promptIndex)
        assertFalse(state.isRecording)
        assertFalse(state.isSaving)
        assertFalse(state.isSaved)
        assertNull(state.savedProfileName)
        assertNull(state.error)
    }

    @Test
    fun `CoachState copy preserves promptIndex when advancePrompt is not set`() {
        // recomputeCoachQualityState() with advancePrompt = false
        // (the discard path) must leave promptIndex untouched so the
        // user lands back on the same Harvard sentence after discard.
        val mid = CoachState(promptIndex = 4, clipCount = 5)
        val afterDiscard = mid.copy(clipCount = 4)
        assertEquals(4, afterDiscard.promptIndex)
    }

    @Test
    fun `CoachState copy can reset error explicitly`() {
        // recomputeCoachQualityState() clears `error` on every
        // success path. Verify the copy semantics support this.
        val errored = CoachState(error = "Recording failed")
        val recovered = errored.copy(error = null, statusMessage = "Captured clip 1.")
        assertNull(recovered.error)
    }

    @Test
    fun `ImportStep enum has all expected values`() {
        val steps = ImportStep.entries
        assertEquals(9, steps.size)
        assertTrue(steps.contains(ImportStep.IDLE))
        assertTrue(steps.contains(ImportStep.EXTRACTING))
        assertTrue(steps.contains(ImportStep.SEGMENTING))
        assertTrue(steps.contains(ImportStep.CLUSTERING))
        assertTrue(steps.contains(ImportStep.SPEAKER_SELECTION))
        assertTrue(steps.contains(ImportStep.READY))
        assertTrue(steps.contains(ImportStep.SAVING))
        assertTrue(steps.contains(ImportStep.COMPLETE))
        assertTrue(steps.contains(ImportStep.ERROR))
    }
}
