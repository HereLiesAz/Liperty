package com.hereliesaz.liperty.voicebox.cloning

import android.content.Context
import android.util.Log
import com.hereliesaz.liperty.voicebox.VoiceState
import org.json.JSONArray
import org.json.JSONObject
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Persists and manages cloned voice identities and rich voice profiles.
 *
 * Two formats are supported:
 * - Legacy `.vstate` files (Java ObjectOutputStream — backward compat)
 * - New `.vprofile` directory per profile (JSON metadata + binary embeddings)
 */
class VoiceStore(private val context: Context) {

    companion object {
        private const val TAG = "VoiceStore"
    }

    private val storeDir = File(context.filesDir, "voices").apply { if (!exists()) mkdirs() }
    private val profileDir = File(context.filesDir, "voice_profiles").apply { if (!exists()) mkdirs() }

    // ── Legacy VoiceState methods ───────────────────────────────────────

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

    fun loadAllVoices(): List<VoiceState> {
        val voices = mutableListOf<VoiceState>()

        // Load legacy .vstate files
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

        // Also surface profiles as VoiceStates
        for (profile in loadAllProfiles()) {
            if (voices.none { it.name == profile.name }) {
                voices.add(profile.toVoiceState())
            }
        }

        return voices
    }

    fun deleteVoice(name: String) {
        File(storeDir, "$name.vstate").delete()
        deleteProfile(name)
    }

    // ── VoiceProfile methods ────────────────────────────────────────────

    /**
     * Saves a [VoiceProfile] as a directory containing:
     * - `meta.json` — profile metadata and per-sample info
     * - `embeddings.bin` — all sample embeddings as raw floats (compact)
     * - `averaged.bin` — the final averaged embedding
     *
     * Also writes a legacy `.vstate` for backward compatibility.
     */
    fun saveProfile(profile: VoiceProfile) {
        val dir = File(profileDir, profile.name).apply { mkdirs() }
        try {
            // 1. Metadata JSON
            val meta = JSONObject().apply {
                put("name", profile.name)
                put("createdAtMs", profile.createdAtMs)
                put("lastUpdatedMs", profile.lastUpdatedMs)
                put("qualityScore", profile.qualityScore.toDouble())
                put("embeddingDim", if (profile.averagedEmbedding.isNotEmpty()) profile.averagedEmbedding.size else 0)
                put("samples", JSONArray().apply {
                    for (sample in profile.samples) {
                        put(JSONObject().apply {
                            put("id", sample.id)
                            put("sourceFileName", sample.sourceFileName)
                            put("durationMs", sample.durationMs)
                            put("snrEstimate", sample.snrEstimate.toDouble())
                            put("addedAtMs", sample.addedAtMs)
                        })
                    }
                })
            }
            File(dir, "meta.json").writeText(meta.toString(2))

            // 2. Per-sample embeddings (binary)
            DataOutputStream(File(dir, "embeddings.bin").outputStream().buffered()).use { dos ->
                for (sample in profile.samples) {
                    for (v in sample.embedding) dos.writeFloat(v)
                }
            }

            // 3. Averaged embedding (binary)
            DataOutputStream(File(dir, "averaged.bin").outputStream().buffered()).use { dos ->
                for (v in profile.averagedEmbedding) dos.writeFloat(v)
            }

            // 4. Legacy VoiceState for backward compat
            saveVoice(profile.toVoiceState())

            Log.i(TAG, "Saved profile '${profile.name}' (${profile.sampleCount} samples, quality=${profile.qualityScore})")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to save profile '${profile.name}'", e)
        }
    }

    fun loadProfile(name: String): VoiceProfile? {
        val dir = File(profileDir, name)
        if (!dir.isDirectory) return null

        return try {
            val metaFile = File(dir, "meta.json")
            if (!metaFile.exists()) return null
            val meta = JSONObject(metaFile.readText())

            val embeddingDim = meta.getInt("embeddingDim")
            val samplesJson = meta.getJSONArray("samples")

            // Read all sample embeddings
            val allEmbeddings = DataInputStream(File(dir, "embeddings.bin").inputStream().buffered()).use { dis ->
                (0 until samplesJson.length()).map {
                    FloatArray(embeddingDim) { dis.readFloat() }
                }
            }

            // Read averaged embedding
            val averaged = DataInputStream(File(dir, "averaged.bin").inputStream().buffered()).use { dis ->
                FloatArray(embeddingDim) { dis.readFloat() }
            }

            val samples = (0 until samplesJson.length()).map { i ->
                val sj = samplesJson.getJSONObject(i)
                VoiceSample(
                    id = sj.getString("id"),
                    sourceFileName = sj.getString("sourceFileName"),
                    durationMs = sj.getLong("durationMs"),
                    embedding = allEmbeddings[i],
                    snrEstimate = sj.getDouble("snrEstimate").toFloat(),
                    addedAtMs = sj.getLong("addedAtMs")
                )
            }

            VoiceProfile(
                name = meta.getString("name"),
                createdAtMs = meta.getLong("createdAtMs"),
                lastUpdatedMs = meta.getLong("lastUpdatedMs"),
                samples = samples,
                averagedEmbedding = averaged,
                qualityScore = meta.getDouble("qualityScore").toFloat()
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load profile '$name'", e)
            null
        }
    }

    fun loadAllProfiles(): List<VoiceProfile> {
        return profileDir.listFiles { f -> f.isDirectory }
            ?.mapNotNull { loadProfile(it.name) }
            ?: emptyList()
    }

    fun deleteProfile(name: String) {
        val dir = File(profileDir, name)
        if (dir.isDirectory) {
            dir.deleteRecursively()
        }
    }
}
