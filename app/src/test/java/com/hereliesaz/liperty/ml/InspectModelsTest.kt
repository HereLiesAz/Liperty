package com.hereliesaz.liperty.ml

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

@RunWith(AndroidJUnit4::class)
@Config(sdk = [34], manifest = Config.NONE)
class InspectModelsTest {

    @Test
    fun inspectAllModels() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val models = listOf(
            "vallr_model.tflite",
            "vsr_lora_model.tflite",
            "ssr_model.tflite",
            "tramba_model.tflite",
            "vsr_model.tflite"
        )

        for (modelName in models) {
            println("--- Inspecting $modelName ---")
            try {
                val engine = TFLiteEngine(context, modelName)
                if (engine.initialize()) {
                    val inputShape = engine.getInputShape(0)
                    val outputShape = engine.getOutputShape(0)
                    println("  Input Shape: ${inputShape.joinToString()}")
                    println("  Output Shape: ${outputShape.joinToString()}")
                } else {
                    println("  Failed to initialize $modelName")
                }
                engine.close()
            } catch (e: Throwable) {
                println("  Error inspecting $modelName: ${e.message}")
            }
        }
    }
}
