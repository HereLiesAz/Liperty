package com.hereliesaz.liperty.voicebox.cloning

import android.content.Context
import com.hereliesaz.liperty.voicebox.VoiceState
import java.io.File
import java.io.ObjectInputStream
import java.io.ObjectOutputStream

/**
 * Persists and manages cloned voice identities.
 */
class VoiceStore(private val context: Context) {

    private val storeDir = File(context.filesDir, "voices").apply { if (!exists()) mkdirs() }

    /**
     * Saves a cloned voice state to internal storage.
     */
    fun saveVoice(voiceState: VoiceState) {
        val file = File(storeDir, "${voiceState.name}.vstate")
        try {
            ObjectOutputStream(file.outputStream()).use { oos ->
                oos.writeObject(voiceState.name)
                oos.writeObject(voiceState.embedding)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Loads all saved voice identities.
     */
    fun loadAllVoices(): List<VoiceState> {
        val voices = mutableListOf<VoiceState>()
        storeDir.listFiles { file -> file.extension == "vstate" }?.forEach { file ->
            try {
                ObjectInputStream(file.inputStream()).use { ois ->
                    val name = ois.readObject() as String
                    val embedding = ois.readObject() as FloatArray
                    voices.add(VoiceState(name, embedding))
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        return voices
    }

    /**
     * Deletes a specific voice identity.
     */
    fun deleteVoice(name: String) {
        File(storeDir, "$name.vstate").delete()
    }
}
