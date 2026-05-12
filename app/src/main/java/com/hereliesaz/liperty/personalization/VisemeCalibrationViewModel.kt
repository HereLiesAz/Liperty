package com.hereliesaz.liperty.personalization

import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

/**
 * State machine for the guided viseme-calibration recording session.
 *
 * The session walks the user through a curated sequence of words drawn
 * from [VisemeConfusionSet]s (loaded by [ConfusionSetLoader]). For each
 * word the user:
 *   1. Sees the word on a prompter.
 *   2. Holds a "record" button while speaking it (typically ~1-2 s).
 *   3. Releases the button; the ViewModel packages the accumulated
 *      lip-ROI frames into a [PairedTrainingRecord] tagged with the
 *      target word + source viseme set, and saves via
 *      [PairedTrainingStore].
 *   4. Advances to the next word.
 *
 * The frame-capture pipeline is OUTSIDE this class — the host
 * (`VisemeCalibrationScreen` / `MainActivity`) is expected to feed
 * 88×88 lip-cropped grayscale [Bitmap]s into [onFrame] while [state]
 * is in [Phase.Recording]. The ViewModel converts them to the
 * float32 NCHW form the LoRA training notebook expects.
 *
 * Why a ViewModel and not just a plain class: we need lifecycle-aware
 * coroutine scope (saving records uses I/O on Dispatchers.IO) AND we
 * need the state to survive configuration changes mid-session so the
 * user doesn't lose half their recordings to a screen rotation.
 */
class VisemeCalibrationViewModel(
    private val store: PairedTrainingStore,
    /**
     * Sets to walk through, in order. Caller picks how many — typically
     * the top-N from [ConfusionSetLoader.load] sorted by usefulness
     * score. With set sizes capped, N=50 sets ≈ 150 words ≈ 5-15 min.
     */
    private val sets: List<VisemeConfusionSet>,
    /**
     * Per-word frame budget. Captures more than this and we trim the
     * leading edge (the first few frames are typically pre-articulation).
     * The on-device inference window is 16 frames, so 32 here gives us
     * a healthy buffer to crop from at training time.
     */
    private val maxFramesPerWord: Int = 32,
    /**
     * Don't save sub-threshold recordings — under N frames means the
     * user tapped instead of held, or face detection dropped out.
     * Defaults to 8 frames (~0.3 s at 25 fps), the lower bound where
     * a single phoneme onset is even visible.
     */
    private val minFramesPerWord: Int = 8,
    /**
     * Crop edge length. 88 matches the SyncVSR / Auto-AVSR input.
     * Configurable for testing.
     */
    private val cropSize: Int = 88,
    /**
     * Normalization constants — must match the encoder ONNX's training
     * preprocessing. Defaults are Chaplin's transforms.VideoTransform.
     */
    private val pixelMean: Float = 0.421f,
    private val pixelStd: Float = 0.165f,
    /**
     * Dispatcher for the disk-write side of [finishRecording]. Production
     * uses [Dispatchers.IO] (real thread pool); tests inject the
     * `StandardTestDispatcher` so `advanceUntilIdle()` actually drives the
     * save coroutine. Without injection the save runs on a real IO thread
     * and never gets dispatched by `runTest`'s scheduler, hanging the test.
     */
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
    /**
     * Dispatcher for the post-save UI-state update. Same rationale as
     * [ioDispatcher]. In production this is [Dispatchers.Main]; tests
     * inject the same `StandardTestDispatcher` they pass for IO, since
     * a single dispatcher is sufficient when both sides of the
     * `withContext` boundary are pure data manipulation.
     */
    private val mainDispatcher: CoroutineDispatcher = Dispatchers.Main,
) : ViewModel() {

    /** Where we are in the [sets] traversal. */
    sealed class Phase {
        /** Waiting for user to press the record button on the current word. */
        data object Idle : Phase()
        /** Frames accumulating into the in-progress recording. */
        data object Recording : Phase()
        /** Save in flight after the user released the button. UI shows a
         *  brief spinner before advancing to the next word. */
        data object Saving : Phase()
        /** All sets done. */
        data object Complete : Phase()
    }

    data class UiState(
        val phase: Phase,
        val currentSetIndex: Int,
        val currentWordIndex: Int,
        val currentSetSize: Int,
        val currentWord: String,
        val currentVisemeSeq: String,
        val totalWords: Int,
        val completedWords: Int,
        /** Frames accumulated for the in-progress recording. Used to
         *  drive a "1.3 s / max 32 frames" indicator in the UI. */
        val recordingFrameCount: Int,
        /** Records saved this session, by id. Lets the UI offer
         *  "undo last" or "show me what I've recorded" without
         *  rescanning the disk. */
        val savedIds: List<String>,
    )

    private val totalWords = sets.sumOf { it.words.size }
    private val flatPlan: List<Pair<Int, Int>> = run {
        // Pre-compute the linear (setIdx, wordIdx) traversal so the UI's
        // "word X of Y" math doesn't need to walk sets every recompose.
        val out = ArrayList<Pair<Int, Int>>(totalWords)
        for ((si, s) in sets.withIndex()) {
            for (wi in s.words.indices) out.add(si to wi)
        }
        out
    }
    private var planIndex = 0

    /** Frames captured for the currently-recording word. Cleared after
     *  each save. Kept as Bitmaps until save time so we don't pay the
     *  float conversion cost on every frame on the UI thread. */
    private val recordingFrames = ArrayList<Bitmap>()

    private val _state = MutableStateFlow(buildIdleState(savedIds = emptyList()))
    val state: StateFlow<UiState> = _state.asStateFlow()

    /**
     * Build a snapshot UiState for the cursor's current position. The
     * savedIds list is passed in explicitly rather than read from
     * `_state.value` because [_state] is initialized FROM this function,
     * so during the initial call there is no `_state` to read yet —
     * that path was the source of an NPE.
     */
    private fun buildIdleState(savedIds: List<String>): UiState {
        if (planIndex >= flatPlan.size) {
            return UiState(
                phase = Phase.Complete,
                currentSetIndex = -1, currentWordIndex = -1, currentSetSize = 0,
                currentWord = "", currentVisemeSeq = "",
                totalWords = totalWords, completedWords = totalWords,
                recordingFrameCount = 0, savedIds = savedIds,
            )
        }
        val (si, wi) = flatPlan[planIndex]
        val set = sets[si]
        return UiState(
            phase = Phase.Idle,
            currentSetIndex = si, currentWordIndex = wi, currentSetSize = set.size,
            currentWord = set.words[wi], currentVisemeSeq = set.visemeSeq,
            totalWords = totalWords, completedWords = planIndex,
            recordingFrameCount = 0,
            savedIds = savedIds,
        )
    }

    /** Start accumulating frames for the current word. */
    fun startRecording() {
        if (_state.value.phase !is Phase.Idle) return
        recordingFrames.clear()
        _state.value = _state.value.copy(phase = Phase.Recording, recordingFrameCount = 0)
    }

    /** Cancel without saving — the user wants to retry. */
    fun cancelRecording() {
        if (_state.value.phase !is Phase.Recording) return
        recordingFrames.clear()
        _state.value = _state.value.copy(phase = Phase.Idle, recordingFrameCount = 0)
    }

    /**
     * Push one mouth-ROI frame into the current recording. No-op unless
     * we're in [Phase.Recording]. Caller owns the bitmap; we copy a
     * reference into the accumulator. (The host pipeline is expected
     * to hand off bitmaps the inference path is done with, since the
     * inference path is paused while calibration is active.)
     *
     * Frames beyond [maxFramesPerWord] are dropped — the user is
     * speaking too long. The UI should hint at this with the frame
     * counter saturating.
     */
    fun onFrame(bitmap: Bitmap) {
        if (_state.value.phase !is Phase.Recording) return
        if (recordingFrames.size >= maxFramesPerWord) return
        recordingFrames.add(bitmap)
        _state.value = _state.value.copy(recordingFrameCount = recordingFrames.size)
    }

    /**
     * User released the record button. Package the frames and save.
     * No-op if fewer than [minFramesPerWord] were captured — the UI
     * should treat that as "retry."
     *
     * Returns true if the save was queued, false if the recording was
     * rejected as too short.
     */
    fun finishRecording(): Boolean {
        if (_state.value.phase !is Phase.Recording) return false
        if (recordingFrames.size < minFramesPerWord) {
            cancelRecording()
            return false
        }
        val cur = _state.value
        val word = cur.currentWord
        val visemeSeq = cur.currentVisemeSeq
        // Snapshot + clear the accumulator so concurrent onFrame calls
        // for the NEXT word can't sneak frames into this one's save.
        val framesSnapshot = ArrayList(recordingFrames)
        recordingFrames.clear()
        _state.value = cur.copy(phase = Phase.Saving)
        viewModelScope.launch(ioDispatcher) {
            val record = buildRecord(word, visemeSeq, framesSnapshot)
            store.save(record)
            withContext(mainDispatcher) {
                planIndex += 1
                val saved = _state.value.savedIds + record.id
                _state.value = buildIdleState(savedIds = saved)
            }
        }
        return true
    }

    /**
     * Skip the current word entirely (no save, just advance). For
     * words the user can't pronounce / doesn't recognise / refuses
     * to speak. The LoRA training corpus is more useful with honest
     * skips than with refusal-noise recordings.
     */
    fun skipCurrent() {
        if (_state.value.phase is Phase.Recording) recordingFrames.clear()
        planIndex += 1
        _state.value = buildIdleState(savedIds = _state.value.savedIds)
    }

    /** Reset to the start of the session. Doesn't delete already-saved
     *  records — those persist for the LoRA pipeline regardless. */
    fun restart() {
        planIndex = 0
        recordingFrames.clear()
        _state.value = buildIdleState(savedIds = _state.value.savedIds)
    }

    private fun buildRecord(
        word: String,
        visemeSeq: String,
        frames: List<Bitmap>,
    ): PairedTrainingRecord {
        val pixelsPerImage = cropSize * cropSize
        val videoFrames = frames.map { bmp -> bitmapToNormalizedFloats(bmp, pixelsPerImage) }
        // Tag the source so the LoRA training notebook can filter for
        // calibration recordings vs. organic-conversation samples.
        val src = "viseme_calibration:${visemeSeq.replace(' ', '_')}:$word"
        return PairedTrainingRecord(
            id = UUID.randomUUID().toString(),
            audioPcm = FloatArray(0),   // Visual-only calibration; no audio captured.
            videoFrames = videoFrames,
            createdAtMs = System.currentTimeMillis(),
            source = src,
            transcript = word,
            transcriptConfidence = 1.0f,   // Ground-truth label, by construction.
        )
    }

    private fun bitmapToNormalizedFloats(bmp: Bitmap, pixelsPerImage: Int): FloatArray {
        val scaled = if (bmp.width != cropSize || bmp.height != cropSize) {
            Bitmap.createScaledBitmap(bmp, cropSize, cropSize, true)
        } else bmp
        val px = IntArray(pixelsPerImage)
        scaled.getPixels(px, 0, cropSize, 0, 0, cropSize, cropSize)
        if (scaled !== bmp) scaled.recycle()
        val out = FloatArray(pixelsPerImage)
        // Rec.601 luma — matches AvHubertSeq2SeqInference and VSRInference.
        for (i in px.indices) {
            val pixel = px[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            val raw = ((r * 76 + g * 150 + b * 30) ushr 8).coerceIn(0, 255)
            out[i] = (raw / 255f - pixelMean) / pixelStd
        }
        return out
    }
}
