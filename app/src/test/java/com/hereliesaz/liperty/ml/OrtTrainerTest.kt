package com.hereliesaz.liperty.ml

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/**
 * Tests for [OrtTrainer] — on-device LoRA training orchestrator.
 *
 * These tests run under Robolectric and cannot load the ONNX Runtime
 * native library, so they verify degradation behavior: artifact checks,
 * graceful failure on initialization, and safe cleanup.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class OrtTrainerTest {

    @Test
    fun artifactsNotAvailableReturnsFalse() {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        val trainer = OrtTrainer(ctx)
        assertFalse(trainer.areArtifactsAvailable())
    }

    @Test
    fun initializeFailsGracefullyWhenArtifactsMissing() = runTest {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        val trainer = OrtTrainer(ctx)
        assertFalse(trainer.initialize())
        assertTrue(trainer.state.value is OrtTrainer.TrainingState.Error)
    }

    @Test
    fun trainReturnsNullWhenNotInitialized() = runTest {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        val trainer = OrtTrainer(ctx)
        val sample = TrainingSample(
            videoFrames = FloatArray(88 * 88),
            numFrames = 1,
            targetFeatures = FloatArray(768),
            numTargetFrames = 1,
        )
        val result = trainer.train(listOf(sample))
        assertTrue(result == null)
        assertTrue(trainer.state.value is OrtTrainer.TrainingState.Error)
    }

    @Test
    fun cancelIsIdempotent() {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        val trainer = OrtTrainer(ctx)
        trainer.cancel()
        trainer.cancel()
        // No exception
    }

    @Test
    fun closeIsIdempotent() {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        val trainer = OrtTrainer(ctx)
        trainer.close()
        trainer.close()
        // No exception
    }

    @Test
    fun closeResetsStateToIdle() {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        val trainer = OrtTrainer(ctx)
        trainer.close()
        assertTrue(trainer.state.value is OrtTrainer.TrainingState.Idle)
    }
}
