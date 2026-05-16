package com.hereliesaz.liperty.voicebox

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Tests for [WavHeader] — the streaming RIFF/WAVE header reader
 * shared by [PocketTTSEngine.readWavFile] and
 * [com.hereliesaz.liperty.voicebox.VoiceViewModel.wavDurationMs].
 *
 * Pure JUnit — no Android dependencies. Builds canonical WAV byte
 * streams in memory and asserts the parsed fields.
 */
class WavHeaderTest {

    @get:Rule
    val tmp = TemporaryFolder()

    // ── synthetic WAV builder ─────────────────────────────────────

    /**
     * Builds a minimal-but-valid RIFF/WAVE byte stream with the
     * given format chunk and a `data` chunk filled with zeros.
     */
    private fun buildWav(
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int,
        dataBytes: Int,
        extraChunkBeforeData: Pair<String, ByteArray>? = null,
    ): ByteArray {
        val fmtSize = 16
        val extraSize = extraChunkBeforeData?.let { 8 + it.second.size } ?: 0
        val riffSize = 4 + (8 + fmtSize) + extraSize + (8 + dataBytes)
        val buf = ByteBuffer.allocate(8 + riffSize).order(ByteOrder.LITTLE_ENDIAN)
        buf.put("RIFF".toByteArray(Charsets.US_ASCII))
        buf.putInt(riffSize)
        buf.put("WAVE".toByteArray(Charsets.US_ASCII))
        // fmt chunk
        buf.put("fmt ".toByteArray(Charsets.US_ASCII))
        buf.putInt(fmtSize)
        buf.putShort(1) // PCM
        buf.putShort(channels.toShort())
        buf.putInt(sampleRate)
        buf.putInt(sampleRate * channels * bitsPerSample / 8) // byte rate
        buf.putShort((channels * bitsPerSample / 8).toShort()) // block align
        buf.putShort(bitsPerSample.toShort())
        // optional spurious chunk
        extraChunkBeforeData?.let { (tag, payload) ->
            buf.put(tag.toByteArray(Charsets.US_ASCII))
            buf.putInt(payload.size)
            buf.put(payload)
        }
        // data chunk
        buf.put("data".toByteArray(Charsets.US_ASCII))
        buf.putInt(dataBytes)
        buf.put(ByteArray(dataBytes))
        return buf.array()
    }

    // ── happy path ─────────────────────────────────────────────────

    @Test
    fun `reads canonical 16 kHz mono 16-bit header`() {
        val bytes = buildWav(sampleRate = 16_000, channels = 1, bitsPerSample = 16, dataBytes = 32_000)
        val h = WavHeader.read(ByteArrayInputStream(bytes))
        assertNotNull(h)
        assertEquals(16_000, h!!.sampleRate)
        assertEquals(1, h.channels)
        assertEquals(16, h.bitsPerSample)
        assertEquals(32_000L, h.dataSizeBytes)
        // data offset = 12 (RIFF/size/WAVE) + 8 (fmt header) + 16 (fmt body) = 36
        assertEquals(36L, h.dataOffsetBytes)
    }

    @Test
    fun `duration math for 1 second of 16 kHz mono 16-bit`() {
        // 16000 frames * 2 bytes/frame = 32000 data bytes → 1000 ms.
        val bytes = buildWav(sampleRate = 16_000, channels = 1, bitsPerSample = 16, dataBytes = 32_000)
        val h = WavHeader.read(ByteArrayInputStream(bytes))!!
        assertEquals(1_000L, h.durationMs)
    }

    @Test
    fun `duration math for 0_5 seconds of 24 kHz stereo 16-bit`() {
        // 12000 frames * 4 bytes/frame = 48000 data bytes → 500 ms.
        val bytes = buildWav(sampleRate = 24_000, channels = 2, bitsPerSample = 16, dataBytes = 48_000)
        val h = WavHeader.read(ByteArrayInputStream(bytes))!!
        assertEquals(2, h.channels)
        assertEquals(500L, h.durationMs)
    }

    // ── chunk skipping ─────────────────────────────────────────────

    @Test
    fun `skips spurious LIST chunk before data`() {
        // Some authoring tools insert LIST/INFO metadata between
        // the fmt chunk and the data chunk — the parser must skip
        // them and still report the right offsets.
        val bytes = buildWav(
            sampleRate = 16_000, channels = 1, bitsPerSample = 16, dataBytes = 32_000,
            extraChunkBeforeData = "LIST" to ByteArray(64) { 0xAA.toByte() },
        )
        val h = WavHeader.read(ByteArrayInputStream(bytes))!!
        assertEquals(16_000, h.sampleRate)
        assertEquals(32_000L, h.dataSizeBytes)
        // Data offset moves forward by the size of the LIST chunk (8 header + 64 payload).
        assertEquals(36L + 8L + 64L, h.dataOffsetBytes)
    }

    @Test
    fun `skips spurious JUNK chunk before data`() {
        val bytes = buildWav(
            sampleRate = 16_000, channels = 1, bitsPerSample = 16, dataBytes = 16_000,
            extraChunkBeforeData = "JUNK" to ByteArray(28),
        )
        val h = WavHeader.read(ByteArrayInputStream(bytes))!!
        assertEquals(16_000L, h.dataSizeBytes)
    }

    // ── rejection ──────────────────────────────────────────────────

    @Test
    fun `returns null on non-RIFF input`() {
        val raw = "totally not a wav file".toByteArray()
        assertNull(WavHeader.read(ByteArrayInputStream(raw)))
    }

    @Test
    fun `returns null on truncated header`() {
        val raw = "RIFF".toByteArray()
        assertNull(WavHeader.read(ByteArrayInputStream(raw)))
    }

    @Test
    fun `returns null when fmt chunk is missing or undersized`() {
        // Build a WAV with a fmt chunk size of 8 (too small for PCM).
        val buf = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN)
        buf.put("RIFF".toByteArray(Charsets.US_ASCII))
        buf.putInt(20)
        buf.put("WAVE".toByteArray(Charsets.US_ASCII))
        buf.put("fmt ".toByteArray(Charsets.US_ASCII))
        buf.putInt(8) // way too small
        buf.put(ByteArray(8))
        assertNull(WavHeader.read(ByteArrayInputStream(buf.array(), 0, buf.position())))
    }

    // ── File overload ─────────────────────────────────────────────

    @Test
    fun `read(File) matches read(InputStream) for the same payload`() {
        val bytes = buildWav(sampleRate = 22_050, channels = 1, bitsPerSample = 16, dataBytes = 44_100)
        val file = tmp.newFile("synthetic.wav")
        file.writeBytes(bytes)
        val viaStream = WavHeader.read(ByteArrayInputStream(bytes))!!
        val viaFile = WavHeader.read(file)!!
        assertEquals(viaStream, viaFile)
    }

    @Test
    fun `read(File) returns null on missing file`() {
        val nonexistent = File(tmp.root, "does_not_exist.wav")
        assertNull(WavHeader.read(nonexistent))
    }
}
