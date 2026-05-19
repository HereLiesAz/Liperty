package com.hereliesaz.liperty.ml

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.hereliesaz.liperty.personalization.PairedTrainingStore
import com.hereliesaz.liperty.personalization.PersonalizationConsentManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Orchestrates the on-device LoRA training flow.
 *
 * Manages prerequisites checking, data preparation, training execution,
 * checkpoint persistence, and inference model export. Exposes a
 * [StateFlow] for the training UI to observe.
 */
class TrainingViewModel(
    private val context: Context,
    private val store: PairedTrainingStore,
    private val consentManager: PersonalizationConsentManager,
) : ViewModel() {

    companion object {
        private const val TAG = "TrainingViewModel"
    }

    sealed class UiState {
        data object CheckingPrerequisites : UiState()
        data class MissingPrerequisites(
            val hasConsent: Boolean,
            val hasArtifacts: Boolean,
            val sampleCount: Int,
        ) : UiState()
        data object Preparing : UiState()
        data class Training(
            val epoch: Int,
            val totalEpochs: Int,
            val step: Int,
            val totalSteps: Int,
            val loss: Float,
            val elapsedMs: Long,
        ) : UiState()
        data object Saving : UiState()
        data class Completed(
            val avgLoss: Float,
            val steps: Int,
            val exportedModelPath: String?,
        ) : UiState()
        data class Error(val message: String) : UiState()
    }

    private val _uiState = MutableStateFlow<UiState>(UiState.CheckingPrerequisites)
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val trainer = OrtTrainer(context)
    private val encoder = AvHubertEncoderSession(context)
    private val preparer by lazy { TrainingDataPreparer(encoder) }

    fun checkPrerequisites() {
        viewModelScope.launch(Dispatchers.Default) {
            val hasConsent = consentManager.hasConsent()
            val hasArtifacts = trainer.areArtifactsAvailable()
            val sampleCount = store.listAll().size
            _uiState.value = UiState.MissingPrerequisites(
                hasConsent, hasArtifacts, sampleCount
            )
        }
    }

    fun startTraining(epochs: Int = 5, learningRate: Float = 1e-4f) {
        viewModelScope.launch(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()
            try {
                // 1. Initialize encoder (for MSE target generation)
                _uiState.value = UiState.Preparing
                if (!encoder.initialize()) {
                    _uiState.value = UiState.Error("Failed to load encoder")
                    return@launch
                }

                // 2. Prepare training data
                val samples = preparer.prepareAll(store)
                if (samples.isEmpty()) {
                    _uiState.value = UiState.Error(
                        "No valid training samples. Record calibration data first."
                    )
                    return@launch
                }
                Log.i(TAG, "Prepared ${samples.size} training samples")

                // 3. Initialize trainer
                if (!trainer.initialize()) {
                    _uiState.value = UiState.Error("Failed to initialize trainer")
                    return@launch
                }

                // 4. Forward trainer state to UI
                launch {
                    trainer.state.collect { ts ->
                        if (ts is OrtTrainer.TrainingState.Training) {
                            _uiState.value = UiState.Training(
                                epoch = ts.epoch,
                                totalEpochs = ts.totalEpochs,
                                step = ts.step,
                                totalSteps = ts.totalSteps,
                                loss = ts.currentLoss,
                                elapsedMs = System.currentTimeMillis() - startTime,
                            )
                        }
                    }
                }

                // 5. Train
                val avgLoss = trainer.train(samples, epochs, learningRate)
                if (avgLoss == null) {
                    _uiState.value = UiState.Error("Training cancelled or failed")
                    return@launch
                }

                // 6. Save checkpoint
                _uiState.value = UiState.Saving
                trainer.saveCheckpoint()

                // 7. Export inference model
                val exportPath = trainer.exportForInference()

                _uiState.value = UiState.Completed(
                    avgLoss = avgLoss,
                    steps = samples.size * epochs,
                    exportedModelPath = exportPath,
                )
                Log.i(TAG, "Training flow complete: avgLoss=$avgLoss, exported=$exportPath")
            } catch (ex: Exception) {
                Log.e(TAG, "Training flow failed", ex)
                _uiState.value = UiState.Error(ex.message ?: "Unknown error")
            }
        }
    }

    fun cancelTraining() {
        trainer.cancel()
    }

    override fun onCleared() {
        trainer.close()
        encoder.close()
    }
}
