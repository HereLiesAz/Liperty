package com.hereliesaz.liperty.personalization

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/**
 * Unit tests for the [VisemeCalibrationViewModel] state machine.
 *
 * `@RunWith(AndroidJUnit4::class) + @Config(sdk=[34])` pattern, NOT
 * Robolectric's default runner — `targetSdk=37` chokes the older
 * default runner (see CLAUDE.md "Robolectric SDK 37 trap"). We need
 * Android because [Bitmap.createBitmap] is used in the
 * frames-to-record conversion path.
 *
 * `Dispatchers.setMain` + `StandardTestDispatcher` lets us drive the
 * ViewModel's save coroutine deterministically — without it,
 * `viewModelScope.launch(Dispatchers.IO)` never runs in a unit test.
 */
@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class VisemeCalibrationViewModelTest {

    @get:Rule val tmp = TemporaryFolder()

    private val testDispatcher = StandardTestDispatcher()
    private val testScope = TestScope(testDispatcher)

    private lateinit var store: PairedTrainingStore

    private val twoSets = listOf(
        VisemeConfusionSet("V2 V1A V4", listOf("bat", "pat", "mat"), score = 10f),
        VisemeConfusionSet("V5 V1B", listOf("the", "they"), score = 10f),
    )

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        store = PairedTrainingStore(tmp.newFolder("recordings"))
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    /** Trivial 88×88 ARGB bitmap. Content doesn't matter for these
     *  tests — the VM only checks frame count and converts pixels
     *  with a deterministic luma formula. */
    private fun stubBitmap(): Bitmap = Bitmap.createBitmap(88, 88, Bitmap.Config.ARGB_8888)

    private fun makeVm(
        sets: List<VisemeConfusionSet> = twoSets,
        minFrames: Int = 8,
        maxFrames: Int = 32,
    ) = VisemeCalibrationViewModel(
        store = store,
        sets = sets,
        maxFramesPerWord = maxFrames,
        minFramesPerWord = minFrames,
        // Inject the test dispatcher for BOTH the I/O and Main sides so
        // `advanceUntilIdle()` drives the save coroutine. Without this,
        // the launch(Dispatchers.IO) call runs on the real thread pool
        // and never gets scheduled by the test scope, hanging tests.
        ioDispatcher = testDispatcher,
        mainDispatcher = testDispatcher,
    )

    @Test
    fun initialStateIsIdleOnFirstWord() = runTest(testDispatcher) {
        val vm = makeVm()
        val s = vm.state.value
        assertEquals(VisemeCalibrationViewModel.Phase.Idle, s.phase)
        assertEquals(0, s.currentSetIndex)
        assertEquals(0, s.currentWordIndex)
        assertEquals("bat", s.currentWord)
        assertEquals("V2 V1A V4", s.currentVisemeSeq)
        // Plan totals: 3 (set 0) + 2 (set 1) = 5 words.
        assertEquals(5, s.totalWords)
        assertEquals(0, s.completedWords)
        assertEquals(0, s.recordingFrameCount)
    }

    @Test
    fun startRecordingTransitionsToRecording() = runTest(testDispatcher) {
        val vm = makeVm()
        vm.startRecording()
        assertEquals(VisemeCalibrationViewModel.Phase.Recording, vm.state.value.phase)
    }

    @Test
    fun framesAccumulateOnlyDuringRecording() = runTest(testDispatcher) {
        val vm = makeVm()
        // Frames pushed during Idle are dropped.
        vm.onFrame(stubBitmap())
        assertEquals(0, vm.state.value.recordingFrameCount)

        vm.startRecording()
        repeat(5) { vm.onFrame(stubBitmap()) }
        assertEquals(5, vm.state.value.recordingFrameCount)
    }

    @Test
    fun finishWithBelowMinFramesRejectsAndStaysOnSameWord() = runTest(testDispatcher) {
        val vm = makeVm(minFrames = 8)
        val wordBefore = vm.state.value.currentWord
        vm.startRecording()
        repeat(3) { vm.onFrame(stubBitmap()) }   // fewer than min
        val accepted = vm.finishRecording()
        assertFalse("save must be rejected", accepted)
        val s = vm.state.value
        assertEquals(VisemeCalibrationViewModel.Phase.Idle, s.phase)
        assertEquals(wordBefore, s.currentWord)
        assertEquals(0, s.completedWords)
        assertTrue("nothing saved", s.savedIds.isEmpty())
    }

    @Test
    fun successfulFinishSavesRecordAndAdvancesWord() = runTest(testDispatcher) {
        val vm = makeVm(minFrames = 4)
        vm.startRecording()
        repeat(10) { vm.onFrame(stubBitmap()) }
        val accepted = vm.finishRecording()
        assertTrue(accepted)
        // Save runs on Dispatchers.IO -> testDispatcher; drain it.
        advanceUntilIdle()

        val s = vm.state.value
        assertEquals(VisemeCalibrationViewModel.Phase.Idle, s.phase)
        // Advanced from "bat" (index 0) to "pat" (index 1).
        assertEquals("pat", s.currentWord)
        assertEquals(1, s.completedWords)
        assertEquals(1, s.savedIds.size)

        // Verify the on-disk record carries the right metadata.
        val saved = store.load(s.savedIds[0])
        assertNotNull(saved)
        assertEquals("bat", saved!!.transcript)
        assertEquals(1f, saved.transcriptConfidence ?: 0f, 1e-6f)
        assertTrue(saved.source.startsWith("viseme_calibration:V2_V1A_V4:bat"))
        assertEquals(10, saved.videoFrames.size)
        assertEquals(88 * 88, saved.videoFrames[0].size)
    }

    @Test
    fun maxFramesPerWordCaps() = runTest(testDispatcher) {
        val vm = makeVm(maxFrames = 4)
        vm.startRecording()
        repeat(10) { vm.onFrame(stubBitmap()) }
        // Frame counter saturates at maxFramesPerWord.
        assertEquals(4, vm.state.value.recordingFrameCount)
    }

    @Test
    fun cancelDiscardsAndStaysOnSameWord() = runTest(testDispatcher) {
        val vm = makeVm()
        val before = vm.state.value.currentWord
        vm.startRecording()
        repeat(20) { vm.onFrame(stubBitmap()) }
        vm.cancelRecording()
        advanceUntilIdle()

        val s = vm.state.value
        assertEquals(VisemeCalibrationViewModel.Phase.Idle, s.phase)
        assertEquals(before, s.currentWord)
        assertEquals(0, s.completedWords)
        assertTrue(s.savedIds.isEmpty())
    }

    @Test
    fun skipAdvancesWithoutSaving() = runTest(testDispatcher) {
        val vm = makeVm()
        vm.skipCurrent()
        val s = vm.state.value
        assertEquals("pat", s.currentWord)
        assertEquals(1, s.completedWords)
        assertTrue(s.savedIds.isEmpty())
    }

    @Test
    fun walkingPastTheLastWordTransitionsToComplete() = runTest(testDispatcher) {
        val vm = makeVm()
        // Skip all 5 — no recordings, just traversal.
        repeat(5) { vm.skipCurrent() }
        advanceUntilIdle()
        val s = vm.state.value
        assertEquals(VisemeCalibrationViewModel.Phase.Complete, s.phase)
        assertEquals(5, s.completedWords)
    }

    @Test
    fun restartReturnsToFirstWordButKeepsSavedRecords() = runTest(testDispatcher) {
        val vm = makeVm(minFrames = 4)
        vm.startRecording()
        repeat(5) { vm.onFrame(stubBitmap()) }
        vm.finishRecording()
        advanceUntilIdle()
        val savedIdBefore = vm.state.value.savedIds.singleOrNull()
        assertNotNull(savedIdBefore)

        vm.restart()
        val s = vm.state.value
        assertEquals(VisemeCalibrationViewModel.Phase.Idle, s.phase)
        assertEquals("bat", s.currentWord)
        assertEquals(0, s.completedWords)
        // The on-disk record persists — restart only resets the
        // ViewModel's in-memory traversal cursor.
        assertNotNull(store.load(savedIdBefore!!))
    }

    @Test
    fun secondRecordingDoesntInheritFirstsFrames() = runTest(testDispatcher) {
        val vm = makeVm(minFrames = 4)
        vm.startRecording()
        repeat(10) { vm.onFrame(stubBitmap()) }
        vm.finishRecording()
        advanceUntilIdle()

        // We're on word 2 ("pat") now. Frame count must be 0 for the
        // new word, even though we accumulated 10 frames a moment ago.
        assertEquals(0, vm.state.value.recordingFrameCount)

        vm.startRecording()
        repeat(7) { vm.onFrame(stubBitmap()) }
        assertEquals(7, vm.state.value.recordingFrameCount)
        vm.finishRecording()
        advanceUntilIdle()

        // Two distinct records saved, one per word.
        val ids = vm.state.value.savedIds
        assertEquals(2, ids.size)
        assertNotEquals(ids[0], ids[1])
        val r0 = store.load(ids[0])!!; val r1 = store.load(ids[1])!!
        assertEquals(10, r0.videoFrames.size)
        assertEquals(7, r1.videoFrames.size)
        assertEquals("bat", r0.transcript)
        assertEquals("pat", r1.transcript)
    }

    @Test
    fun recordingFramesOutsideRecordingPhaseAreDropped() = runTest(testDispatcher) {
        // Defensive: even if the camera pipeline is sloppy about
        // unhooking the callback, frames pushed during Idle/Saving/
        // Complete phases must be no-ops.
        val vm = makeVm(minFrames = 4)
        vm.onFrame(stubBitmap())
        assertEquals(0, vm.state.value.recordingFrameCount)

        vm.startRecording()
        repeat(5) { vm.onFrame(stubBitmap()) }
        vm.finishRecording()
        // BEFORE advanceUntilIdle: we're in Phase.Saving. Frames now
        // should also be dropped because the recordingFrames list was
        // already snapshotted and cleared.
        vm.onFrame(stubBitmap())
        advanceUntilIdle()
        assertEquals(0, vm.state.value.recordingFrameCount)
    }

    @Test
    fun emptySetListLandsImmediatelyInComplete() = runTest(testDispatcher) {
        val vm = makeVm(sets = emptyList())
        assertEquals(VisemeCalibrationViewModel.Phase.Complete, vm.state.value.phase)
        assertEquals(0, vm.state.value.totalWords)
    }

    @Test
    fun underlyingStoreReceivesNormalizedFrames() = runTest(testDispatcher) {
        // Sanity check: each frame in a saved record is a FloatArray
        // of length cropSize*cropSize, with values in the expected
        // Rec.601 + mean/std normalisation range. We're not asserting
        // exact pixel values (the stub bitmap is all-zeros + alpha=0)
        // but the SHAPE must be right or the LoRA notebook can't read
        // the records.
        val vm = makeVm(minFrames = 4)
        vm.startRecording()
        repeat(6) { vm.onFrame(stubBitmap()) }
        vm.finishRecording()
        advanceUntilIdle()
        val saved = store.load(vm.state.value.savedIds[0])!!
        assertEquals(6, saved.videoFrames.size)
        for (frame in saved.videoFrames) {
            assertEquals(88 * 88, frame.size)
            // All-zeros input -> luma 0 -> (0/255 - 0.421) / 0.165 ≈ -2.55.
            assertEquals(-2.551f, frame[0], 1e-2f)
        }
    }
}
