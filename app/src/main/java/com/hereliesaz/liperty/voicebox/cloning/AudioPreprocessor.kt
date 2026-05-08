package com.hereliesaz.liperty.voicebox.cloning

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.util.Log
import com.hereliesaz.liperty.dsp.VibraPhoneDSP
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Preprocesses audio/video files for voice cloning.
 *
 * Pipeline:
 *  1. Extract audio track from any media file (MediaExtractor + MediaCodec)
 *  2. Resample to 16 kHz mono PCM
 *  3. Voice Activity Detection (energy + zero-crossing rate)
 *  4. Segment into 1–10 s speech chunks
 *  5. Optional denoising via spectral subtraction
 *
 * All output stays in RAM as [FloatArray] — no intermediate files are written to disk.
 */
class AudioPreprocessor(private val context: Context) {

    companion object {
        private const val TAG = "AudioPreprocessor"
        private const val TARGET_SAMPLE_RATE = 16_000
        private const val MIN_SEGMENT_SAMPLES = TARGET_SAMPLE_RATE * 1      // 1 s
        private const val MAX_SEGMENT_SAMPLES = TARGET_SAMPLE_RATE * 10     // 10 s
        private const val MERGE_GAP_SAMPLES = (TARGET_SAMPLE_RATE * 0.3).toInt() // 300 ms
        private const val VAD_FRAME_SIZE = 512
        private const val VAD_HOP_SIZE = 256
    }

    data class SpeechSegment(
        val pcmData: FloatArray,
        val startTimeMs: Long,
        val durationMs: Long,
        val sourceFileName: String,
        val rmsEnergy: Float,
        val snrEstimate: Float
    )

    data class PreprocessResult(
        val segments: List<SpeechSegment>,
        val totalDurationMs: Long,
        val speechDurationMs: Long,
        val sourceFormat: String
    )

    /**
     * Processes a content URI (audio or video) into speech segments.
     * Runs blocking I/O — call from [kotlinx.coroutines.Dispatchers.IO].
     */
    fun processUri(uri: Uri, fileName: String = "unknown"): PreprocessResult {
        val (pcm, format) = extractAudioFromUri(uri)
        return buildResult(pcm, format, fileName)
    }

    /**
     * Processes a raw WAV file (e.g. from [com.hereliesaz.liperty.voicebox.recording.VoiceRecorder]).
     */
    fun processWavFile(file: File): PreprocessResult {
        val pcm = loadWav(file)
        return buildResult(pcm, "audio/wav", file.name)
    }

    // ── Public utilities (useful for testing) ───────────────────────────

    /**
     * Energy + zero-crossing-rate Voice Activity Detection.
     * @return list of (startSample, endSample) pairs identifying speech regions.
     */
    fun detectSpeechRegions(
        pcm: FloatArray,
        energyThreshold: Float = 0.005f,
        zcrThreshold: Float = 0.3f
    ): List<Pair<Int, Int>> {
        if (pcm.isEmpty()) return emptyList()

        val frameCount = max(0, (pcm.size - VAD_FRAME_SIZE) / VAD_HOP_SIZE + 1)
        if (frameCount == 0) return emptyList()

        // Per-frame energy and ZCR
        val energies = FloatArray(frameCount)
        val zcrs = FloatArray(frameCount)
        for (f in 0 until frameCount) {
            val offset = f * VAD_HOP_SIZE
            var energy = 0f
            var zc = 0
            for (i in 0 until VAD_FRAME_SIZE) {
                val idx = offset + i
                if (idx >= pcm.size) break
                energy += pcm[idx] * pcm[idx]
                if (i > 0 && idx > 0) {
                    val prev = pcm[idx - 1]
                    val curr = pcm[idx]
                    if ((prev >= 0f && curr < 0f) || (prev < 0f && curr >= 0f)) zc++
                }
            }
            energies[f] = energy / VAD_FRAME_SIZE
            zcrs[f] = zc.toFloat() / VAD_FRAME_SIZE
        }

        // Classify frames: speech if energy above threshold OR (moderate energy AND low ZCR for voiced)
        val isSpeech = BooleanArray(frameCount) { f ->
            energies[f] > energyThreshold ||
                    (energies[f] > energyThreshold * 0.3f && zcrs[f] < zcrThreshold)
        }

        // Convert frame-level decisions to sample-level regions
        val regions = mutableListOf<Pair<Int, Int>>()
        var regionStart = -1
        for (f in isSpeech.indices) {
            if (isSpeech[f] && regionStart < 0) {
                regionStart = f * VAD_HOP_SIZE
            } else if (!isSpeech[f] && regionStart >= 0) {
                regions.add(regionStart to min(f * VAD_HOP_SIZE + VAD_FRAME_SIZE, pcm.size))
                regionStart = -1
            }
        }
        if (regionStart >= 0) {
            regions.add(regionStart to pcm.size)
        }
        return regions
    }

    /**
     * Merges close segments, splits long ones, discards short ones.
     */
    fun refineSpeechSegments(
        regions: List<Pair<Int, Int>>,
        totalSamples: Int,
        minSamples: Int = MIN_SEGMENT_SAMPLES,
        maxSamples: Int = MAX_SEGMENT_SAMPLES,
        mergeGapSamples: Int = MERGE_GAP_SAMPLES
    ): List<Pair<Int, Int>> {
        if (regions.isEmpty()) return emptyList()

        // 1. Merge segments separated by less than mergeGapSamples
        val merged = mutableListOf<Pair<Int, Int>>()
        var (curStart, curEnd) = regions[0]
        for (i in 1 until regions.size) {
            val (nextStart, nextEnd) = regions[i]
            if (nextStart - curEnd <= mergeGapSamples) {
                curEnd = nextEnd
            } else {
                merged.add(curStart to curEnd)
                curStart = nextStart
                curEnd = nextEnd
            }
        }
        merged.add(curStart to curEnd)

        // 2. Split segments longer than maxSamples, discard shorter than minSamples
        val result = mutableListOf<Pair<Int, Int>>()
        for ((start, end) in merged) {
            val len = end - start
            if (len < minSamples) continue
            if (len <= maxSamples) {
                result.add(start to min(end, totalSamples))
            } else {
                // Split into roughly equal chunks
                var s = start
                while (s < end) {
                    val e = min(s + maxSamples, end)
                    if (e - s >= minSamples) {
                        result.add(s to min(e, totalSamples))
                    }
                    s = e
                }
            }
        }
        return result
    }

    /**
     * Resamples PCM from [inputRate] to [outputRate] using linear interpolation.
     */
    fun resample(input: FloatArray, inputRate: Int, outputRate: Int = TARGET_SAMPLE_RATE): FloatArray {
        if (inputRate == outputRate) return input
        val ratio = inputRate.toDouble() / outputRate
        val outputLen = (input.size / ratio).toInt()
        if (outputLen <= 0) return floatArrayOf()
        val output = FloatArray(outputLen)
        for (i in output.indices) {
            val srcIdx = i * ratio
            val idx0 = srcIdx.toInt().coerceIn(0, input.size - 1)
            val idx1 = (idx0 + 1).coerceIn(0, input.size - 1)
            val frac = (srcIdx - idx0).toFloat()
            output[i] = input[idx0] * (1f - frac) + input[idx1] * frac
        }
        return output
    }

    // ── Private implementation ──────────────────────────────────────────

    private fun buildResult(pcm: FloatArray, format: String, fileName: String): PreprocessResult {
        val totalDurationMs = (pcm.size.toLong() * 1000) / TARGET_SAMPLE_RATE

        // VAD
        val rawRegions = detectSpeechRegions(pcm)
        val regions = refineSpeechSegments(rawRegions, pcm.size)

        // Estimate noise from non-speech regions for denoising
        val noiseProfile = estimateNoiseProfile(pcm, regions)

        // Build segments
        val segments = regions.map { (start, end) ->
            val raw = pcm.copyOfRange(start, end)
            val denoised = denoise(raw, noiseProfile)
            val rms = computeRms(denoised)
            val snr = estimateSnr(rms, noiseProfile)

            SpeechSegment(
                pcmData = denoised,
                startTimeMs = (start.toLong() * 1000) / TARGET_SAMPLE_RATE,
                durationMs = ((end - start).toLong() * 1000) / TARGET_SAMPLE_RATE,
                sourceFileName = fileName,
                rmsEnergy = rms,
                snrEstimate = snr
            )
        }

        val speechDurationMs = segments.sumOf { it.durationMs }

        Log.i(TAG, "Processed $fileName: ${segments.size} segments, ${speechDurationMs}ms speech / ${totalDurationMs}ms total")

        return PreprocessResult(
            segments = segments,
            totalDurationMs = totalDurationMs,
            speechDurationMs = speechDurationMs,
            sourceFormat = format
        )
    }

    /**
     * Extracts audio from any media URI using MediaExtractor + MediaCodec.
     * Returns 16 kHz mono PCM float samples and the source MIME type.
     */
    private fun extractAudioFromUri(uri: Uri): Pair<FloatArray, String> {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(context, uri, null)

            // Find the first audio track
            var audioTrackIndex = -1
            var audioFormat: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("audio/")) {
                    audioTrackIndex = i
                    audioFormat = format
                    break
                }
            }

            if (audioTrackIndex < 0 || audioFormat == null) {
                Log.w(TAG, "No audio track found in URI")
                return floatArrayOf() to "unknown"
            }

            val mime = audioFormat.getString(MediaFormat.KEY_MIME) ?: "audio/raw"
            val sampleRate = audioFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channelCount = audioFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

            extractor.selectTrack(audioTrackIndex)

            // Decode using MediaCodec
            val codec = MediaCodec.createDecoderByType(mime)
            codec.configure(audioFormat, null, null, 0)
            codec.start()

            val rawPcm = mutableListOf<Float>()
            val bufferInfo = MediaCodec.BufferInfo()
            var inputDone = false

            while (true) {
                // Feed input
                if (!inputDone) {
                    val inputIdx = codec.dequeueInputBuffer(10_000)
                    if (inputIdx >= 0) {
                        val inputBuf = codec.getInputBuffer(inputIdx) ?: continue
                        val sampleSize = extractor.readSampleData(inputBuf, 0)
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inputIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inputIdx, 0, sampleSize, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                // Drain output
                val outputIdx = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                if (outputIdx >= 0) {
                    val outputBuf = codec.getOutputBuffer(outputIdx)
                    if (outputBuf != null && bufferInfo.size > 0) {
                        outputBuf.order(ByteOrder.LITTLE_ENDIAN)
                        outputBuf.position(bufferInfo.offset)
                        val shortBuf = outputBuf.asShortBuffer()
                        val numShorts = bufferInfo.size / 2
                        for (i in 0 until numShorts) {
                            val sample = shortBuf.get()
                            // Down-mix to mono by taking every Nth sample (N = channelCount)
                            if (channelCount <= 1 || i % channelCount == 0) {
                                rawPcm.add(sample.toFloat() / Short.MAX_VALUE)
                            }
                        }
                    }
                    codec.releaseOutputBuffer(outputIdx, false)

                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                } else if (outputIdx == MediaCodec.INFO_TRY_AGAIN_LATER && inputDone) {
                    break
                }
            }

            codec.stop()
            codec.release()

            val pcmArray = rawPcm.toFloatArray()
            val resampled = resample(pcmArray, sampleRate, TARGET_SAMPLE_RATE)
            return resampled to mime

        } finally {
            extractor.release()
        }
    }

    /**
     * Loads a WAV file into 16 kHz mono float PCM.
     */
    private fun loadWav(file: File): FloatArray {
        val bytes = file.readBytes()
        if (bytes.size < 44) return floatArrayOf()

        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // Parse WAV header
        buf.position(22)
        val channels = buf.short.toInt()
        val sampleRate = buf.int
        buf.position(34)
        val bitsPerSample = buf.short.toInt()

        // Find data chunk
        buf.position(36)
        while (buf.remaining() >= 8) {
            val chunkId = ByteArray(4)
            buf.get(chunkId)
            val chunkSize = buf.int
            if (String(chunkId) == "data") {
                val numSamples = chunkSize / (bitsPerSample / 8)
                val pcm = FloatArray(numSamples / channels)
                var outIdx = 0
                for (i in 0 until numSamples) {
                    val sample = if (bitsPerSample == 16) {
                        buf.short.toFloat() / Short.MAX_VALUE
                    } else {
                        (buf.get().toInt() and 0xFF).toFloat() / 128f - 1f
                    }
                    // Take first channel only for mono downmix
                    if (i % channels == 0 && outIdx < pcm.size) {
                        pcm[outIdx++] = sample
                    }
                }
                return resample(pcm.copyOf(outIdx), sampleRate, TARGET_SAMPLE_RATE)
            } else {
                // Skip non-data chunks
                if (buf.remaining() >= chunkSize) buf.position(buf.position() + chunkSize)
                else break
            }
        }
        return floatArrayOf()
    }

    /**
     * Estimates a noise profile from non-speech regions of the audio.
     */
    private fun estimateNoiseProfile(pcm: FloatArray, speechRegions: List<Pair<Int, Int>>): FloatArray {
        // Collect non-speech samples
        val nonSpeech = mutableListOf<Float>()
        var pos = 0
        for ((start, end) in speechRegions.sortedBy { it.first }) {
            for (i in pos until min(start, pcm.size)) {
                nonSpeech.add(pcm[i])
            }
            pos = end
        }
        for (i in pos until pcm.size) {
            nonSpeech.add(pcm[i])
        }

        if (nonSpeech.size < VAD_FRAME_SIZE) {
            // Not enough non-speech data; use a conservative floor
            return FloatArray(VAD_FRAME_SIZE / 2) { 0.002f }
        }

        // Compute average magnitude spectrum of non-speech
        val frameCount = min(10, nonSpeech.size / VAD_FRAME_SIZE)
        val profileSize = VAD_FRAME_SIZE / 2
        val profile = FloatArray(profileSize)
        for (f in 0 until frameCount) {
            val offset = f * VAD_FRAME_SIZE
            for (i in 0 until profileSize) {
                val idx = offset + i
                if (idx < nonSpeech.size) {
                    profile[i] += abs(nonSpeech[idx]) / frameCount
                }
            }
        }
        return profile
    }

    /**
     * Applies spectral subtraction for denoising.
     * Falls back to simple scaling if native DSP is unavailable.
     */
    private fun denoise(segment: FloatArray, noiseProfile: FloatArray): FloatArray {
        if (segment.size < VAD_FRAME_SIZE) return segment

        return try {
            // Process in FRAME_SIZE windows via VibraPhoneDSP
            val dsp = VibraPhoneDSP()
            val output = segment.copyOf()
            val frameSize = VibraPhoneDSP.FRAME_SIZE
            var offset = 0
            while (offset + frameSize <= output.size) {
                val frame = output.copyOfRange(offset, offset + frameSize)
                dsp.spectralSubtraction(frame, noiseProfile)
                System.arraycopy(frame, 0, output, offset, frameSize)
                offset += frameSize
            }
            output
        } catch (e: UnsatisfiedLinkError) {
            // Native library not available — return undenoised audio
            Log.w(TAG, "Native DSP unavailable, skipping denoising")
            segment
        } catch (e: Exception) {
            Log.w(TAG, "Denoising failed, returning raw segment", e)
            segment
        }
    }

    private fun computeRms(pcm: FloatArray): Float {
        if (pcm.isEmpty()) return 0f
        var sum = 0.0
        for (s in pcm) sum += s * s
        return sqrt(sum / pcm.size).toFloat()
    }

    private fun estimateSnr(signalRms: Float, noiseProfile: FloatArray): Float {
        val noiseRms = if (noiseProfile.isNotEmpty()) {
            var sum = 0f
            for (n in noiseProfile) sum += n * n
            sqrt(sum / noiseProfile.size)
        } else 0.001f

        if (noiseRms < 1e-10f) return 40f // Very quiet noise floor
        val snr = 20f * kotlin.math.log10(signalRms / noiseRms)
        return snr.coerceIn(0f, 60f)
    }
}
