package com.hereliesaz.liperty.personalization

import org.json.JSONObject
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

/**
 * On-disk storage for [PairedTrainingRecord]s. One directory holds all
 * records as triples of files:
 *   - `<id>.audio` — binary float32 PCM (big-endian, count-prefixed)
 *   - `<id>.video` — binary frames (count-prefixed; each frame = float32 row-major)
 *   - `<id>.json`  — metadata (id, createdAtMs, source, transcript, etc.)
 *
 * The format is intentionally simple — no Room/SQLite/etc. — so the
 * "delete all biometric data" pathway from Settings is just
 * `directory.deleteRecursively()` and is trivially auditable.
 *
 * Thread-safety: the store is a thin wrapper around the filesystem; the
 * filesystem itself serializes concurrent operations on the same file.
 * Concurrent `save()` of distinct ids is safe; concurrent `save()`+`load()`
 * of the SAME id is not guaranteed atomic — callers should synchronize at
 * a higher level if they need that.
 */
class PairedTrainingStore(private val rootDir: File) {

    init {
        if (!rootDir.exists()) rootDir.mkdirs()
    }

    fun save(rec: PairedTrainingRecord) {
        val audioFile = audioFile(rec.id)
        val videoFile = videoFile(rec.id)
        val metaFile = metaFile(rec.id)

        DataOutputStream(audioFile.outputStream().buffered()).use { out ->
            out.writeInt(rec.audioPcm.size)
            for (v in rec.audioPcm) out.writeFloat(v)
        }

        DataOutputStream(videoFile.outputStream().buffered()).use { out ->
            out.writeInt(rec.videoFrames.size)
            // Encode the size of a single frame once — the writer assumes
            // all frames have the same length, which the encoder pipeline
            // guarantees (always 88x88 = 7744 floats).
            val frameLen = rec.videoFrames.firstOrNull()?.size ?: 0
            out.writeInt(frameLen)
            for (frame in rec.videoFrames) {
                require(frame.size == frameLen) {
                    "frame size mismatch: expected $frameLen, got ${frame.size}"
                }
                for (v in frame) out.writeFloat(v)
            }
        }

        val meta = JSONObject().apply {
            put("id", rec.id)
            put("createdAtMs", rec.createdAtMs)
            put("source", rec.source)
            rec.transcript?.let { put("transcript", it) }
            rec.transcriptConfidence?.let { put("transcriptConfidence", it.toDouble()) }
        }
        metaFile.writeText(meta.toString())
    }

    fun load(id: String): PairedTrainingRecord? {
        val audioFile = audioFile(id)
        val videoFile = videoFile(id)
        val metaFile = metaFile(id)
        if (!audioFile.exists() || !videoFile.exists() || !metaFile.exists()) return null

        val audio = DataInputStream(audioFile.inputStream().buffered()).use { input ->
            val n = input.readInt()
            FloatArray(n).also { for (i in 0 until n) it[i] = input.readFloat() }
        }

        val frames = DataInputStream(videoFile.inputStream().buffered()).use { input ->
            val nFrames = input.readInt()
            val frameLen = input.readInt()
            val list = ArrayList<FloatArray>(nFrames)
            for (f in 0 until nFrames) {
                val frame = FloatArray(frameLen)
                for (i in 0 until frameLen) frame[i] = input.readFloat()
                list.add(frame)
            }
            list
        }

        val meta = JSONObject(metaFile.readText())
        return PairedTrainingRecord(
            id = meta.getString("id"),
            audioPcm = audio,
            videoFrames = frames,
            createdAtMs = meta.getLong("createdAtMs"),
            source = meta.getString("source"),
            transcript = if (meta.has("transcript")) meta.getString("transcript") else null,
            transcriptConfidence = if (meta.has("transcriptConfidence"))
                meta.getDouble("transcriptConfidence").toFloat() else null,
        )
    }

    fun listAll(): List<String> {
        val files = rootDir.listFiles().orEmpty()
        return files.mapNotNull { f ->
            val name = f.name
            if (name.endsWith(".json")) name.removeSuffix(".json") else null
        }.sorted()
    }

    fun delete(id: String) {
        audioFile(id).delete()
        videoFile(id).delete()
        metaFile(id).delete()
    }

    /**
     * The Settings "delete all biometric data" path (GDPR/BIPA right-to-erasure).
     * Recursively removes the entire store directory — including any stray
     * subdirectory or partially-written record — then recreates the empty root,
     * so nothing biometric can be left behind. Returns true only if the tree was
     * fully removed; a false result means some file could not be deleted (e.g.
     * locked) and the caller should surface a failure rather than report success.
     */
    fun deleteAll(): Boolean {
        val removed = rootDir.deleteRecursively()
        rootDir.mkdirs()
        return removed
    }

    fun totalBytes(): Long =
        rootDir.listFiles().orEmpty().sumOf { it.length() }

    private fun audioFile(id: String) = File(rootDir, "$id.audio")
    private fun videoFile(id: String) = File(rootDir, "$id.video")
    private fun metaFile(id: String) = File(rootDir, "$id.json")
}
