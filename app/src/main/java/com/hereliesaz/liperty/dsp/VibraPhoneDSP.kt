package com.hereliesaz.liperty.dsp

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.PI

/**
 * Implements Digital Signal Processing algorithms for Laryngeal Sensing via Back-EMF (VibraPhone).
 * Based on research: "VibraPhone: Listening through a Vibration Motor" (MobiSys 2016).
 *
 * Key pipeline steps:
 * 1. Spectral Subtraction (Noise Reduction)
 * 2. Frequency Domain Equalization (Formant Correction)
 * 3. Voice Source Expansion (High-frequency reconstruction)
 */
class VibraPhoneDSP {

    companion object {
        const val SAMPLE_RATE = 16000
        const val FRAME_SIZE = 512
        const val HOP_SIZE = 256
        private const val EXCITED_HARMONICS_SCALE = 0.5f
    }

    /**
     * Applies Spectral Subtraction to remove stationary noise (e.g., electrical hum).
     * @param inputSignal Raw float array of audio samples.
     * @param noiseProfile Pre-calculated magnitude spectrum of the noise (silence).
     */
    fun spectralSubtraction(inputSignal: FloatArray, noiseProfile: FloatArray): FloatArray {
        // This is a simplified time-domain implementation wrapper.
        // In a real scenario, we'd do STFT -> Subtract Magnitude -> ISTFT.
        // For efficiency on mobile, we assume inputSignal is a single frame or small buffer.
        
        // 1. FFT
        val complexSpectrum = fft(inputSignal)
        val magnitudes = calculateMagnitudes(complexSpectrum)
        val phases = calculatePhases(complexSpectrum)

        // 2. Subtract Noise Spectrum
        val cleanedMagnitudes = FloatArray(magnitudes.size)
        for (i in magnitudes.indices) {
            val noiseMag = if (i < noiseProfile.size) noiseProfile[i] else 0f
            // Oversubtraction factor (alpha) typically 2.0 for aggressive reduction
            val alpha = 2.0f
            val beta = 0.01f // Spectral floor
            
            val subtracted = magnitudes[i] - (alpha * noiseMag)
            cleanedMagnitudes[i] = max(subtracted, beta * magnitudes[i])
        }

        // 3. Reconstruct Complex Spectrum
        val cleanedComplex = reconstructComplex(cleanedMagnitudes, phases)

        // 4. Inverse FFT
        return ifft(cleanedComplex)
    }

    /**
     * Applies Frequency Domain Equalization to correct formant shifting caused by the sensor's mass.
     * The physical mass of the sensor (LRA or phone chassis) acts as a low-pass filter and shifts resonances.
     */
    fun frequencyDomainEqualization(inputSignal: FloatArray): FloatArray {
        // Apply an inverse filter approximating the sensor's transfer function.
        // In SSI mode with Artificial Larynx, the carrier buzz is the sound source.
        // We boost speech-frequency bands (300Hz - 3kHz) where the user's vocal tract
        // modulation is concentrated.
        
        val complexSpectrum = fft(inputSignal)
        val n = complexSpectrum.size / 2
        
        for (i in 0 until n) {
            val freq = i * SAMPLE_RATE.toFloat() / FRAME_SIZE
            
            // Boost speech modulation band (approx 300Hz to 3.5kHz)
            // This emphasizes the modulation created by mouth movements over the raw carrier.
            val gain = if (freq in 300f..3500f) 2.5f else 1.0f
            
            complexSpectrum[2 * i] *= gain
            complexSpectrum[2 * i + 1] *= gain
        }
        
        return ifft(complexSpectrum)
    }

    /**
     * Step 4: Voice Source Expansion via LPC Vocoding (Formant Extrapolation).
     *
     * Replaces the previous dummy spectral-folding approach with a proper
     * Linear Predictive Coding (LPC) vocoder:
     *  1. Estimate the vocal-tract filter (LPC all-pole model, order 12).
     *  2. Detect fundamental pitch (F0) via autocorrelation.
     *  3. Generate an appropriate excitation: glottal-pulse train (voiced) or white noise (unvoiced).
     *  4. Re-synthesize through the LPC filter, restoring intelligible formant structure.
     */
    fun voiceSourceExpansion(inputSignal: FloatArray): FloatArray {
        val lpcOrder = 12
        val lpcCoeffs = computeLPC(inputSignal, lpcOrder)

        val f0 = estimatePitch(inputSignal)
        val excitation = if (f0 > 60f) {
            generateGlottalPulses(inputSignal.size, f0)
        } else {
            generateWhiteNoise(inputSignal.size)
        }

        // Match RMS energy of excitation to input so the synthesised output has the same loudness
        val inputRms = sqrt(inputSignal.map { it * it }.average().toFloat().coerceAtLeast(1e-10f))
        val excRms  = sqrt(excitation.map  { it * it }.average().toFloat().coerceAtLeast(1e-10f))
        val scaled  = excitation.map { it * (inputRms / excRms) }.toFloatArray()

        return allPoleFilter(scaled, lpcCoeffs)
    }

    // ── LPC helpers ────────────────────────────────────────────────────────

    /**
     * Estimates LPC coefficients of [order] using the Levinson-Durbin recursion.
     * Returns the prediction-error coefficients a[1..order] (a[0] = 1 is implicit).
     */
    private fun computeLPC(signal: FloatArray, order: Int): FloatArray {
        // Autocorrelation lags 0..order
        val r = FloatArray(order + 1)
        for (lag in 0..order) {
            var acc = 0.0
            for (j in 0 until signal.size - lag) acc += signal[j] * signal[j + lag]
            r[lag] = acc.toFloat()
        }
        if (r[0] == 0f) return FloatArray(order)

        // Levinson-Durbin
        val a = FloatArray(order + 1).also { it[0] = 1f }
        var err = r[0]
        for (m in 1..order) {
            var lambda = 0f
            for (j in 1 until m) lambda += a[j] * r[m - j]
            val km = -(r[m] + lambda) / err
            val prev = a.copyOf()
            a[m] = km
            for (j in 1 until m) a[j] = prev[j] + km * prev[m - j]
            err *= (1f - km * km)
            if (err <= 0f) break
        }
        return a.drop(1).toFloatArray()   // return a[1..order]
    }

    /**
     * Autocorrelation-based pitch estimator.
     * Searches the lag range corresponding to 60–500 Hz and returns F0 in Hz,
     * or 0 if no clear periodicity is found (unvoiced).
     */
    private fun estimatePitch(signal: FloatArray): Float {
        val minLag = SAMPLE_RATE / 500
        val maxLag = minOf(SAMPLE_RATE / 60, signal.size / 2 - 1)
        if (maxLag <= minLag) return 0f

        var bestCorr = 0f
        var bestLag  = 0
        for (lag in minLag..maxLag) {
            var corr = 0f
            for (i in 0 until signal.size - lag) corr += signal[i] * signal[i + lag]
            if (corr > bestCorr) { bestCorr = corr; bestLag = lag }
        }
        return if (bestLag > 0) SAMPLE_RATE.toFloat() / bestLag else 0f
    }

    /** Rosenberg-model glottal pulse train for voiced excitation. */
    private fun generateGlottalPulses(size: Int, f0: Float): FloatArray {
        val period    = (SAMPLE_RATE / f0).toInt().coerceAtLeast(1)
        val openPhase = (period * 0.4f).toInt().coerceAtLeast(1)
        return FloatArray(size) { i ->
            val phase = i % period
            if (phase < openPhase) sin(PI * phase / openPhase).toFloat() else 0f
        }
    }

    /** White-noise burst for unvoiced excitation. */
    private fun generateWhiteNoise(size: Int): FloatArray {
        val rng = java.util.Random()
        return FloatArray(size) { rng.nextFloat() * 2f - 1f }
    }

    /**
     * LPC all-pole synthesis filter: y[n] = x[n] - sum_k(a[k]*y[n-k-1]).
     * Output is hard-clipped to [-1, 1] to prevent runaway resonances.
     */
    private fun allPoleFilter(excitation: FloatArray, lpcCoeffs: FloatArray): FloatArray {
        val output = FloatArray(excitation.size)
        val order  = lpcCoeffs.size
        for (n in output.indices) {
            var acc = excitation[n].toDouble()
            for (k in 0 until minOf(order, n)) acc -= lpcCoeffs[k] * output[n - k - 1]
            output[n] = acc.toFloat().coerceIn(-1f, 1f)
        }
        return output
    }

    /**
     * TRAMBA High-frequency bandwidth expansion model.
     * Restores attenuated high-frequency components from BCMs using non-linear excitation
     * and spectral shaping, acting as a lightweight DSP substitute for the full neural model.
     */
    fun trambaBandwidthExpansion(inputSignal: FloatArray): FloatArray {
        // 1. Non-linear excitation to generate missing harmonics
        val excitedSignal = inputSignal.map { it * kotlin.math.abs(it) }.toFloatArray()

        val complexSpectrum = fft(inputSignal)
        val excitedSpectrum = fft(excitedSignal)

        val n = complexSpectrum.size / 2
        // BCMs typically attenuate heavily above 3-4kHz. We'll reconstruct above 3kHz.
        val cutoffIdx = (3000 * FRAME_SIZE / SAMPLE_RATE).toInt()

        if (cutoffIdx < n) {
            for (i in cutoffIdx until n) {
                // Add excited high frequencies with a scaling factor
                complexSpectrum[2 * i] += excitedSpectrum[2 * i] * 0.3f
                complexSpectrum[2 * i + 1] += excitedSpectrum[2 * i + 1] * 0.3f
            }
        }
        return ifft(complexSpectrum)
    }

    // --- FFT Helpers (Radix-2 Cooley-Tukey) ---
    
    /**
     * Performs a forward FFT on the input signal.
     * @return Interleaved float array [Re, Im, Re, Im...] of size 2*input.size
     */
    private fun fft(input: FloatArray): FloatArray {
        val n = input.size
        // Ensure n is power of 2
        if (n and (n - 1) != 0) {
            throw IllegalArgumentException("FFT size must be a power of 2. Got: $n")
        }
        
        val complex = FloatArray(n * 2)
        for (i in 0 until n) {
            complex[2 * i] = input[i]
            complex[2 * i + 1] = 0f
        }
        
        fftInPlace(complex, n, false)
        return complex
    }

    /**
     * Performs an inverse FFT.
     * @param complexInput Interleaved float array [Re, Im...]
     */
    private fun ifft(complexInput: FloatArray): FloatArray {
        val n = complexInput.size / 2
        val work = complexInput.copyOf()
        
        fftInPlace(work, n, true)
        
        val result = FloatArray(n)
        for (i in 0 until n) {
            result[i] = work[2 * i] / n // Scaling
        }
        return result
    }

    private fun fftInPlace(data: FloatArray, n: Int, inverse: Boolean) {
        // Bit-reversal permutation
        var j = 0
        for (i in 0 until n) {
            if (i < j) {
                val tempRe = data[2 * i]
                val tempIm = data[2 * i + 1]
                data[2 * i] = data[2 * j]
                data[2 * i + 1] = data[2 * j + 1]
                data[2 * j] = tempRe
                data[2 * j + 1] = tempIm
            }
            var m = n shr 1
            while (m >= 1 && j >= m) {
                j -= m
                m = m shr 1
            }
            j += m
        }

        // Butterfly stages
        var len = 2
        while (len <= n) {
            val ang = 2.0 * Math.PI / len * (if (inverse) -1 else 1)
            val wlenRe = cos(ang).toFloat()
            val wlenIm = sin(ang).toFloat()
            
            var i = 0
            while (i < n) {
                var wRe = 1f
                var wIm = 0f
                for (k in 0 until len / 2) {
                    val uIdx = 2 * (i + k)
                    val vIdx = 2 * (i + k + len / 2)
                    
                    val uRe = data[uIdx]
                    val uIm = data[uIdx + 1]
                    val vRe = data[vIdx] * wRe - data[vIdx + 1] * wIm
                    val vIm = data[vIdx] * wIm + data[vIdx + 1] * wRe
                    
                    data[uIdx] = uRe + vRe
                    data[uIdx + 1] = uIm + vIm
                    data[vIdx] = uRe - vRe
                    data[vIdx + 1] = uIm - vIm
                    
                    val nextWRe = wRe * wlenRe - wIm * wlenIm
                    wIm = wRe * wlenIm + wIm * wlenRe
                    wRe = nextWRe
                }
                i += len
            }
            len *= 2
        }
    }

    private fun calculateMagnitudes(complex: FloatArray): FloatArray {
        val n = complex.size / 2
        val mags = FloatArray(n)
        for (i in 0 until n) {
            val re = complex[2 * i]
            val im = complex[2 * i + 1]
            mags[i] = sqrt(re * re + im * im)
        }
        return mags
    }

    private fun calculatePhases(complex: FloatArray): FloatArray {
        val n = complex.size / 2
        val phases = FloatArray(n)
        for (i in 0 until n) {
            phases[i] = atan2(complex[2 * i + 1], complex[2 * i])
        }
        return phases
    }
    
    private fun reconstructComplex(mags: FloatArray, phases: FloatArray): FloatArray {
        val n = mags.size
        val complex = FloatArray(n * 2)
        for (i in 0 until n) {
            complex[2 * i] = mags[i] * cos(phases[i])
            complex[2 * i + 1] = mags[i] * sin(phases[i])
        }
        return complex
    }
}
