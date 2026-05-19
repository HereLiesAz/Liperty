package com.hereliesaz.liperty.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtTrainingSession
import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.FloatBuffer

/**
 * On-device LoRA training using ONNX Runtime Training Session.
 *
 * Wraps [OrtTrainingSession] with Liperty-specific concerns:
 * - Training artifact resolution (filesDir / assets)
 * - Progress/cancellation reporting via [StateFlow]
 * - Checkpoint persistence to a user-specific directory
 * - Post-training inference model export
 *
 * Thread safety: all training operations run on [Dispatchers.Default].
 * Callers must not invoke train/save/export concurrently.
 */
class OrtTrainer(private val context: Context) {

    companion object {
        private const val TAG = "OrtTrainer"
        const val TRAINING_MODEL = "training/training_model.onnx"
        const val EVAL_MODEL = "training/eval_model.onnx"
        const val OPTIMIZER_MODEL = "training/optimizer_model.onnx"
        const val CHECKPOINT_DIR = "training/checkpoint"
        const val USER_CHECKPOINT_DIR = "personal_checkpoint"
        const val EXPORTED_INFERENCE_MODEL = "personal_inference.onnx"
    }

    sealed class TrainingState {
        data object Idle : TrainingState()
        data object Initializing : TrainingState()
        data class Training(
            val epoch: Int,
            val totalEpochs: Int,
            val step: Int,
            val totalSteps: Int,
            val currentLoss: Float,
        ) : TrainingState()
        data class Completed(val finalLoss: Float, val totalSteps: Int) : TrainingState()
        data class Error(val message: String) : TrainingState()
    }

    private val _state = MutableStateFlow<TrainingState>(TrainingState.Idle)
    val state: StateFlow<TrainingState> = _state.asStateFlow()

    private var env: OrtEnvironment? = null
    private var trainingSession: OrtTrainingSession? = null
    @Volatile private var cancelRequested = false

    /** Check if all training artifacts are available on disk. */
    fun areArtifactsAvailable(): Boolean {
        return listOf(TRAINING_MODEL, EVAL_MODEL, OPTIMIZER_MODEL).all {
            resolveFile(it).let { f -> f.exists() && f.length() > 100 }
        } && resolveFile(CHECKPOINT_DIR).let { it.exists() && it.isDirectory }
    }

    /**
     * Initialize the training session. Must be called before [train].
     * Returns true if initialization succeeded.
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.Default) {
        _state.value = TrainingState.Initializing
        try {
            val e = OrtEnvironment.getEnvironment()

            val checkpointPath = resolveCheckpointPath()
            val trainPath = ensureOnDisk(TRAINING_MODEL)
            val evalPath = ensureOnDisk(EVAL_MODEL)
            val optimizerPath = ensureOnDisk(OPTIMIZER_MODEL)

            val session = e.createTrainingSession(
                checkpointPath, trainPath, evalPath, optimizerPath
            )

            env = e
            trainingSession = session
            _state.value = TrainingState.Idle

            Log.i(TAG, "Training session initialized")
            Log.i(TAG, "  train inputs: ${session.trainInputNames}")
            Log.i(TAG, "  train outputs: ${session.trainOutputNames}")
            true
        } catch (ex: Exception) {
            Log.e(TAG, "Failed to initialize training session", ex)
            _state.value = TrainingState.Error(ex.message ?: "Unknown error")
            false
        }
    }

    /**
     * Run training loop over the provided data.
     *
     * @param samples Training samples (video frames + target features)
     * @param epochs Number of passes over the data
     * @param learningRate Initial learning rate for AdamW
     * @return Final average loss, or null if cancelled/failed
     */
    suspend fun train(
        samples: List<TrainingSample>,
        epochs: Int = 5,
        learningRate: Float = 1e-4f,
    ): Float? = withContext(Dispatchers.Default) {
        val session = trainingSession ?: run {
            _state.value = TrainingState.Error("Not initialized")
            return@withContext null
        }
        val e = env ?: return@withContext null

        cancelRequested = false
        session.setLearningRate(learningRate)

        val totalSteps = epochs * samples.size
        var stepCount = 0
        var runningLoss = 0f

        for (epoch in 0 until epochs) {
            val shuffled = samples.shuffled()
            for ((idx, sample) in shuffled.withIndex()) {
                if (cancelRequested) {
                    Log.i(TAG, "Training cancelled at epoch $epoch step $idx")
                    _state.value = TrainingState.Idle
                    return@withContext null
                }

                val loss = trainStep(e, session, sample)
                runningLoss += loss
                stepCount++

                _state.value = TrainingState.Training(
                    epoch = epoch,
                    totalEpochs = epochs,
                    step = stepCount,
                    totalSteps = totalSteps,
                    currentLoss = loss,
                )

                if (stepCount % 10 == 0) {
                    Log.d(TAG, "Step $stepCount/$totalSteps loss=$loss " +
                        "avgLoss=${runningLoss / stepCount}")
                }
            }
        }

        val avgLoss = if (stepCount > 0) runningLoss / stepCount else 0f
        _state.value = TrainingState.Completed(avgLoss, stepCount)
        Log.i(TAG, "Training complete: $stepCount steps, avgLoss=$avgLoss")
        avgLoss
    }

    /** Request cancellation of the current training loop. */
    fun cancel() {
        cancelRequested = true
    }

    /** Save the current checkpoint to the user's personal directory. */
    suspend fun saveCheckpoint() = withContext(Dispatchers.Default) {
        val session = trainingSession ?: return@withContext
        val dir = File(context.filesDir, USER_CHECKPOINT_DIR)
        if (!dir.exists()) dir.mkdirs()
        session.saveCheckpoint(dir.toPath(), false)
        Log.i(TAG, "Checkpoint saved to ${dir.absolutePath}")
    }

    /**
     * Export a standalone inference-only ONNX from the trained checkpoint.
     * Returns the path to the exported model, or null on failure.
     */
    suspend fun exportForInference(): String? = withContext(Dispatchers.Default) {
        val session = trainingSession ?: return@withContext null
        try {
            val outPath = File(context.filesDir, EXPORTED_INFERENCE_MODEL)
            session.exportModelForInference(
                outPath.toPath(),
                arrayOf("adapted_features")
            )
            Log.i(TAG, "Exported inference model to ${outPath.absolutePath}" +
                " (${outPath.length()} bytes)")
            outPath.absolutePath
        } catch (ex: Exception) {
            Log.e(TAG, "Export failed", ex)
            null
        }
    }

    fun close() {
        trainingSession?.close()
        trainingSession = null
        env = null
        _state.value = TrainingState.Idle
    }

    // ── Internal ──────────────────────────────────────────────────────

    private fun trainStep(
        env: OrtEnvironment,
        session: OrtTrainingSession,
        sample: TrainingSample,
    ): Float {
        val inputNames = session.trainInputNames.toList()
        val inputs = mutableMapOf<String, OnnxTensor>()

        try {
            // Video input: (1, 1, T, 88, 88) float32
            val videoShape = longArrayOf(
                1L, 1L, sample.numFrames.toLong(), 88L, 88L
            )
            inputs[inputNames[0]] = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(sample.videoFrames), videoShape
            )

            // MSE target: (1, T_out, 768) float32
            val targetShape = longArrayOf(
                1L, sample.numTargetFrames.toLong(), 768L
            )
            inputs[inputNames[1]] = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(sample.targetFeatures), targetShape
            )

            // Forward + loss + backward
            val result = session.trainStep(inputs)
            val lossTensor = result.get(0) as OnnxTensor
            val lossValue = lossTensor.floatBuffer.get(0)
            result.close()

            // Optimizer step + gradient reset
            session.optimizerStep()
            session.lazyResetGrad()

            return lossValue
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    /**
     * Resolve the checkpoint path. If the user has a previously saved
     * checkpoint, use it (resume training). Otherwise use the initial
     * checkpoint from artifacts.
     */
    private fun resolveCheckpointPath(): String {
        val userCk = File(context.filesDir, USER_CHECKPOINT_DIR)
        if (userCk.exists() && userCk.isDirectory &&
            userCk.listFiles()?.isNotEmpty() == true
        ) {
            Log.i(TAG, "Resuming from user checkpoint at ${userCk.absolutePath}")
            return userCk.absolutePath
        }
        return ensureOnDisk(CHECKPOINT_DIR)
    }

    private fun resolveFile(name: String): File = File(context.filesDir, name)

    /**
     * Ensures a training artifact is available on disk.
     * Checks filesDir first, then tries to copy from bundled assets.
     */
    private fun ensureOnDisk(name: String): String {
        val dst = File(context.filesDir, name)

        // Already on disk (runtime download or previous asset copy)
        if (dst.exists()) {
            if (dst.isDirectory || dst.length() > 100) return dst.absolutePath
        }

        // Try copying from assets (dev builds with setup_libs.sh)
        try {
            val assetList = context.assets.list(name)
            if (assetList != null && assetList.isNotEmpty()) {
                // It's a directory in assets — copy each child
                dst.mkdirs()
                for (child in assetList) {
                    val childDst = File(dst, child)
                    if (!childDst.exists()) {
                        context.assets.open("$name/$child").use { inp ->
                            childDst.outputStream().use { out -> inp.copyTo(out) }
                        }
                    }
                }
                Log.i(TAG, "Copied asset directory $name to ${dst.absolutePath}")
                return dst.absolutePath
            }
        } catch (_: Exception) { }

        try {
            context.assets.open(name).use { inp ->
                dst.parentFile?.mkdirs()
                dst.outputStream().use { out -> inp.copyTo(out) }
            }
            Log.i(TAG, "Copied asset $name to ${dst.absolutePath}")
            return dst.absolutePath
        } catch (_: Exception) { }

        throw IllegalStateException(
            "Training artifact '$name' not found. " +
            "Download personalization models from Settings."
        )
    }
}
