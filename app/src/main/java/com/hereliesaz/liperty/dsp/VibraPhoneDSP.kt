package com.hereliesaz.liperty.dsp

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

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
     * Step 4: Voice Source Expansion (Formant Extrapolation)
     * Reconstructs missing high-frequency harmonics (>2kHz) based on F0 and F1/F2.
     */
    fun voiceSourceExpansion(inputSignal: FloatArray): FloatArray {
        // Implementation note: Using a non-linear excitation model to
        // recreate high-frequency harmonics (>2kHz).
        
        // 1. Apply non-linear excitation directly to the time domain inputSignal (x * |x|)
        val excitedSignal = inputSignal.map { it * kotlin.math.abs(it) }.toFloatArray()

        // 2. FFT of original and excited signal
        val complexSpectrum = fft(inputSignal)
        val excitedSpectrum = fft(excitedSignal)
        
        val n = complexSpectrum.size / 2
        val cutoffIdx = (2000 * FRAME_SIZE / SAMPLE_RATE).toInt()

        // 3. High-pass filter: replace/add to original spectrum above cutoff
        if (cutoffIdx < n) {
            for (i in cutoffIdx until n) {
                // Scale down the excited harmonics to blend naturally
                complexSpectrum[2 * i] = excitedSpectrum[2 * i] * EXCITED_HARMONICS_SCALE
                complexSpectrum[2 * i + 1] = excitedSpectrum[2 * i + 1] * EXCITED_HARMONICS_SCALE
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
