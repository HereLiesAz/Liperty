package com.hereliesaz.liperty.voicebox

import java.io.DataInputStream
import java.io.EOFException
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Shared WAV (RIFF) metadata reader.
 *
 * Streams just the header and chunk boundary bytes rather than
 * loading the entire file — important for arbitrary user-supplied
 * reference audio (long voicemails, video extracts), where the
 * previous full-file `readBytes()` approach risked OOM on multi-MB
 * inputs.
 *
 * The on-disk WAV format is a "RIFF" container of one or more
 * little-endian chunks: `RIFF<size>WAVE` followed by an "fmt "
 * chunk (sample rate / channels / bits per sample) and a "data"
 * chunk (the PCM samples). Other chunks (LIST/INFO, JUNK, etc.)
 * may appear between fmt and data and must be skipped over.
 *
 * Both [PocketTTSEngine.readWavFile] and
 * [com.hereliesaz.liperty.voicebox.VoiceViewModel] consume this
 * to avoid drift between the two parsers.
 */
data class WavHeader(
    /** Sample rate in Hz from the `fmt ` chunk. */
    val sampleRate: Int,
    /** Channel count from the `fmt ` chunk. */
    val channels: Int,
    /** Bits per sample from the `fmt ` chunk. */
    val bitsPerSample: Int,
    /** Byte offset of the first sample (just past the `data` chunk header). */
    val dataOffsetBytes: Long,
    /** Declared size in bytes of the data chunk. */
    val dataSizeBytes: Long,
) {
    /** Duration of the audio data in milliseconds. */
    val durationMs: Long
        get() {
            val bytesPerSample = (bitsPerSample / 8).coerceAtLeast(1)
            val frames = dataSizeBytes / bytesPerSample.toLong() / channels.coerceAtLeast(1).toLong()
            return frames * 1000L / sampleRate.coerceAtLeast(1).toLong()
        }

    companion object {
        /**
         * Reads the header of [file] without loading the full payload.
         * Returns null if [file] is not a parseable PCM RIFF/WAVE
         * stream (callers can then fall back to assuming raw PCM
         * at a known rate / channel count).
         *
         * Streams at most a few hundred bytes (RIFF header + fmt
         * chunk + any intermediate non-data chunks' headers).
         */
        fun read(file: File): WavHeader? = try {
            file.inputStream().buffered().use { read(it) }
        } catch (_: IOException) {
            null
        }

        /**
         * Same as [read] but takes a pre-opened stream so callers
         * that already have a [java.io.FileInputStream] (or want
         * to inject a fixture in tests) don't have to reopen.
         * Caller owns closing the stream.
         */
        fun read(stream: InputStream): WavHeader? {
            val dis = DataInputStream(stream)
            return try {
                // RIFF header.
                val riff = readAscii(dis, 4)
                if (riff != "RIFF") return null
                readLE32(dis) // chunkSize, ignored
                val wave = readAscii(dis, 4)
                if (wave != "WAVE") return null

                var bytesSeen = 12L
                var sampleRate = 0
                var channels = 0
                var bitsPerSample = 0
                var dataSize = 0L
                var dataOffset = -1L

                while (true) {
                    val tag = try { readAscii(dis, 4) } catch (_: EOFException) { return null }
                    val size = readLE32(dis).toLong() and 0xFFFF_FFFFL
                    bytesSeen += 8
                    when (tag) {
                        "fmt " -> {
                            // Standard PCM fmt chunk: 16 bytes; some
                            // encoders write 18 or 40. Read what we
                            // need then skip any trailing extension.
                            if (size < 16) return null
                            readLE16(dis) // audio format (1 = PCM)
                            channels = readLE16(dis)
                            sampleRate = readLE32(dis)
                            readLE32(dis) // byte rate
                            readLE16(dis) // block align
                            bitsPerSample = readLE16(dis)
                            val consumed = 16L
                            if (size > consumed) skipFully(dis, size - consumed)
                            bytesSeen += size
                        }
                        "data" -> {
                            dataSize = size
                            dataOffset = bytesSeen
                            // Done — header parse complete. Don't
                            // skip the data; the caller may want to
                            // stream it. We've reached the data
                            // boundary, that's enough.
                            return if (sampleRate <= 0 || channels <= 0) {
                                null
                            } else {
                                WavHeader(
                                    sampleRate = sampleRate,
                                    channels = channels,
                                    bitsPerSample = bitsPerSample,
                                    dataOffsetBytes = dataOffset,
                                    dataSizeBytes = dataSize,
                                )
                            }
                        }
                        else -> {
                            // Unknown chunk (LIST/INFO, JUNK, etc.) — skip.
                            skipFully(dis, size)
                            bytesSeen += size
                        }
                    }
                }
                @Suppress("UNREACHABLE_CODE")
                null
            } catch (_: EOFException) {
                null
            } catch (_: IOException) {
                null
            }
        }

        // ── primitives ─────────────────────────────────────────────

        private fun readAscii(dis: DataInputStream, n: Int): String {
            val buf = ByteArray(n)
            dis.readFully(buf)
            return String(buf, Charsets.US_ASCII)
        }

        private fun readLE16(dis: DataInputStream): Int {
            val buf = ByteArray(2)
            dis.readFully(buf)
            return ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xFFFF
        }

        private fun readLE32(dis: DataInputStream): Int {
            val buf = ByteArray(4)
            dis.readFully(buf)
            return ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).int
        }

        /**
         * DataInputStream.skip() is allowed to skip fewer bytes than
         * requested; loop until done or EOF.
         */
        private fun skipFully(dis: DataInputStream, count: Long) {
            var remaining = count
            while (remaining > 0) {
                val skipped = dis.skip(remaining)
                if (skipped <= 0) break
                remaining -= skipped
            }
        }
    }
}
