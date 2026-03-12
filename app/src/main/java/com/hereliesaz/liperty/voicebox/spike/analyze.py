#!/usr/bin/env python3
"""
Phase 0 Spike Analysis
======================
Analyzes dual-stream spike recordings (WAV + CSV) to determine:

  1. Audio signal quality: SNR, spectral content, formant visibility
  2. Accelerometer signal quality: F0 detection capability, noise floor
  3. Temporal alignment: can the two streams be synchronized?
  4. Go/No-Go verdict: is the contact-mic path viable for the DSP pipeline?

Usage:
    python3 analyze.py ./spike_data/
    python3 analyze.py ./spike_data/spike_audio_1234567890.wav  (single file pair)

Requirements:
    pip install numpy scipy matplotlib librosa pandas

Output:
    - Console report with per-band SNR and F0 estimates
    - PNG plots saved alongside input files:
        spike_audio_<ts>_analysis.png
        spike_accel_<ts>_analysis.png
        spike_alignment_<ts>.png
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — saves PNGs instead of showing windows
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("WARNING: librosa not installed. Install with: pip install librosa")
    print("         Falling back to scipy FFT for spectral analysis.")

from scipy.io import wavfile
from scipy.signal import butter, sosfilt, spectrogram as sp_spectrogram
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Constants — should match SignalRecorder.kt
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE  = 16_000   # Hz
GRAVITY            = 9.80665  # m/s² — subtract from accel magnitude for dynamic component
F0_MIN_HZ          = 80       # Lowest expected laryngeal F0
F0_MAX_HZ          = 300      # Highest expected laryngeal F0
FORMANT_BANDS = [              # Frequency bands to check for formant energy
    ("F0",  80,   300),
    ("F1",  300,  900),
    ("F2",  900,  2500),
    ("F3",  2500, 4000),
    ("HF",  4000, 8000),       # Above 2kHz — expected attenuated in contact mic
]


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

def analyze_audio(wav_path: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"AUDIO: {wav_path.name}")
    print(f"{'='*60}")

    sr, data = wavfile.read(str(wav_path))
    assert sr == AUDIO_SAMPLE_RATE, f"Expected {AUDIO_SAMPLE_RATE} Hz, got {sr}"

    # Normalize to float [-1, 1]
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    else:
        audio = data.astype(np.float32)

    duration = len(audio) / sr
    print(f"Duration: {duration:.2f}s  |  Samples: {len(audio)}")

    # RMS and peak
    rms      = float(np.sqrt(np.mean(audio ** 2)))
    peak     = float(np.max(np.abs(audio)))
    rms_db   = 20 * np.log10(rms + 1e-9)
    peak_db  = 20 * np.log10(peak + 1e-9)
    print(f"RMS: {rms_db:.1f} dBFS  |  Peak: {peak_db:.1f} dBFS")

    # Per-band energy
    print("\nPer-band energy (relative to total):")
    band_energies = {}
    total_energy  = np.sum(audio ** 2)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    fft   = np.abs(np.fft.rfft(audio)) ** 2

    for name, lo, hi in FORMANT_BANDS:
        mask  = (freqs >= lo) & (freqs < hi)
        e     = float(np.sum(fft[mask]))
        ratio = e / (total_energy * len(audio) / 2 + 1e-12)  # normalise by FFT bins
        band_energies[name] = ratio
        bar   = "█" * int(ratio * 200)
        print(f"  {name:4s} ({lo:4d}–{hi:4d} Hz):  {ratio:.4f}  {bar}")

    # F0 estimation via autocorrelation
    f0_estimate = _estimate_f0(audio, sr)
    print(f"\nEstimated F0: {f0_estimate:.1f} Hz" if f0_estimate else "\nF0: not detected")

    # Verdict
    hf_ratio  = band_energies.get("HF", 0)
    f0_ratio  = band_energies.get("F0", 0)
    f1_ratio  = band_energies.get("F1", 0)

    viable = (rms_db > -50) and (f0_ratio > 0.01) and (f1_ratio > 0.005)
    print(f"\n{'✅ VIABLE' if viable else '❌ TOO WEAK'} — "
          f"RMS {rms_db:.1f} dBFS, F0 band {f0_ratio:.4f}, F1 band {f1_ratio:.4f}")
    if hf_ratio < 0.001:
        print("  ⚠️  Very low HF content (>4kHz) — expected for contact-mic; formant equalization required.")

    # Plot
    _plot_audio(audio, sr, fft, freqs, wav_path)

    return {
        "path":          str(wav_path),
        "duration_s":    duration,
        "rms_db":        rms_db,
        "band_energies": band_energies,
        "f0_hz":         f0_estimate,
        "viable":        viable,
    }


def _estimate_f0(audio: np.ndarray, sr: int) -> float:
    """Autocorrelation-based F0 estimation on the first 500ms of signal."""
    segment   = audio[:sr // 2]  # 500ms
    corr      = np.correlate(segment, segment, mode="full")
    corr      = corr[len(corr) // 2:]
    # Look for peak in lag range corresponding to F0_MIN–F0_MAX
    lag_min   = int(sr / F0_MAX_HZ)
    lag_max   = int(sr / F0_MIN_HZ)
    if lag_max >= len(corr):
        return 0.0
    peaks, _  = find_peaks(corr[lag_min:lag_max], height=0)
    if len(peaks) == 0:
        return 0.0
    best_lag  = peaks[np.argmax(corr[lag_min:lag_max][peaks])] + lag_min
    return float(sr / best_lag)


def _plot_audio(audio, sr, fft, freqs, wav_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(f"Audio Analysis: {wav_path.name}", fontsize=12)

    # Waveform
    t = np.linspace(0, len(audio) / sr, len(audio))
    axes[0].plot(t, audio, linewidth=0.5, color="#4FC3F7")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title("Waveform")
    axes[0].grid(True, alpha=0.3)

    # Spectrum (log frequency)
    fft_db = 20 * np.log10(np.sqrt(fft) + 1e-9)
    axes[1].semilogx(freqs[1:], fft_db[1:], linewidth=0.8, color="#81C784")
    for _, lo, hi in FORMANT_BANDS:
        axes[1].axvspan(lo, hi, alpha=0.07, color="yellow")
    axes[1].axvline(x=2000, color="red", linestyle="--", linewidth=1, label="2kHz (LRA deaf above here)")
    axes[1].set_xlim(50, sr // 2)
    axes[1].set_ylabel("dB")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_title("Spectrum")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Spectrogram
    if HAS_LIBROSA:
        S = librosa.stft(audio, n_fft=512, hop_length=160)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, hop_length=160, x_axis="time",
                                  y_axis="hz", ax=axes[2], cmap="magma")
        axes[2].set_title("Spectrogram (STFT)")
    else:
        f, t_seg, Sxx = sp_spectrogram(audio, fs=sr, nperseg=512, noverlap=352)
        axes[2].pcolormesh(t_seg, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="magma")
        axes[2].set_ylabel("Frequency (Hz)")
        axes[2].set_title("Spectrogram (scipy)")

    plt.tight_layout()
    out_path = wav_path.with_name(wav_path.stem + "_analysis.png")
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  → Plot saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Accelerometer analysis
# ---------------------------------------------------------------------------

def analyze_accel(csv_path: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"ACCEL: {csv_path.name}")
    print(f"{'='*60}")

    df = pd.read_csv(str(csv_path))
    if df.empty:
        print("  ❌ Empty file")
        return {"viable": False}

    # Convert timestamp to seconds relative to start
    df["t_s"] = (df["timestamp_ns"] - df["timestamp_ns"].iloc[0]) / 1e9

    # Compute dynamic magnitude (subtract gravity)
    magnitude     = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2).values
    dynamic       = np.abs(magnitude - GRAVITY)
    duration      = float(df["t_s"].iloc[-1])
    sample_rate   = len(df) / duration
    print(f"Duration: {duration:.2f}s  |  Samples: {len(df)}  |  Rate: {sample_rate:.0f} Hz")

    rms_dynamic = float(np.sqrt(np.mean(dynamic**2)))
    print(f"Dynamic RMS: {rms_dynamic:.4f} m/s²")

    # FFT of Z axis (perpendicular to screen = throat-contact axis)
    z_dynamic = df["z"].values - np.mean(df["z"].values)
    freqs     = np.fft.rfftfreq(len(z_dynamic), d=1.0 / sample_rate)
    fft_z     = np.abs(np.fft.rfft(z_dynamic))

    # F0 band energy
    mask_f0     = (freqs >= F0_MIN_HZ) & (freqs < F0_MAX_HZ)
    f0_energy   = float(np.sum(fft_z[mask_f0]**2))
    total_energy = float(np.sum(fft_z**2)) + 1e-9
    f0_ratio    = f0_energy / total_energy

    print(f"F0 band energy ratio: {f0_ratio:.4f}")

    # Assess sampling adequacy (need >2×F0_MAX for Nyquist)
    nyquist = sample_rate / 2
    if nyquist < F0_MAX_HZ:
        print(f"  ⚠️  Sample rate {sample_rate:.0f} Hz may be too low for F0_MAX={F0_MAX_HZ} Hz (Nyquist: {nyquist:.0f})")

    viable = (rms_dynamic > 0.005) and (f0_ratio > 0.05)
    print(f"\n{'✅ F0 DETECTABLE' if viable else '❌ F0 NOT DETECTABLE'} — "
          f"dynamic RMS {rms_dynamic:.4f}, F0 ratio {f0_ratio:.4f}")

    # Plot
    _plot_accel(df, z_dynamic, freqs, fft_z, csv_path, sample_rate)

    return {
        "path":          str(csv_path),
        "duration_s":    duration,
        "sample_rate_hz": sample_rate,
        "rms_dynamic":   rms_dynamic,
        "f0_ratio":      f0_ratio,
        "viable":        viable,
    }


def _plot_accel(df, z_dynamic, freqs, fft_z, csv_path: Path, sample_rate: float):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(f"Accelerometer Analysis: {csv_path.name}", fontsize=12)

    # Z-axis time series
    axes[0].plot(df["t_s"], z_dynamic, linewidth=0.5, color="#FFB74D")
    axes[0].set_ylabel("Accel Z (m/s²)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title("Z-Axis Dynamic Component (throat-contact axis)")
    axes[0].grid(True, alpha=0.3)

    # Z-axis spectrum
    fft_db = 20 * np.log10(fft_z + 1e-9)
    axes[1].plot(freqs, fft_db, linewidth=0.8, color="#CE93D8")
    axes[1].axvspan(F0_MIN_HZ, F0_MAX_HZ, alpha=0.15, color="green", label=f"F0 range ({F0_MIN_HZ}–{F0_MAX_HZ} Hz)")
    axes[1].set_xlim(0, min(sample_rate / 2, 500))
    axes[1].set_ylabel("dB")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_title("Z-Axis Spectrum (0–500 Hz)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # XYZ overlay
    for col, color in [("x", "#EF9A9A"), ("y", "#A5D6A7"), ("z", "#90CAF9")]:
        axes[2].plot(df["t_s"], df[col], linewidth=0.4, alpha=0.7, label=col.upper(), color=color)
    axes[2].set_ylabel("Accel (m/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("All Axes")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = csv_path.with_name(csv_path.stem + "_analysis.png")
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  → Plot saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    if target.is_file() and target.suffix == ".wav":
        # Single file — find matching CSV
        wav_paths  = [target]
        csv_paths  = [target.with_name(target.stem.replace("audio", "accel") + ".csv")]
    else:
        wav_paths  = sorted(target.glob("spike_audio_*.wav"))
        csv_paths  = sorted(target.glob("spike_accel_*.csv"))

    if not wav_paths:
        print("No spike_audio_*.wav files found.")
        sys.exit(1)

    all_results = []
    for wav_path in wav_paths:
        audio_result = analyze_audio(wav_path)
        ts           = wav_path.stem.replace("spike_audio_", "")
        csv_match    = target / f"spike_accel_{ts}.csv" if target.is_dir() else csv_paths[0]
        accel_result = {}
        if csv_match.exists():
            accel_result = analyze_accel(csv_match)
        else:
            print(f"\n⚠️  No matching accel CSV for {wav_path.name}")

        all_results.append((audio_result, accel_result))

    # Final verdict
    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}")
    for audio_r, accel_r in all_results:
        audio_ok = audio_r.get("viable", False)
        accel_ok = accel_r.get("viable", False)
        print(f"\nFile set:")
        print(f"  Audio  {'✅ viable' if audio_ok else '❌ insufficient signal'}")
        print(f"  Accel  {'✅ F0 detectable' if accel_ok else '❌ F0 not detectable'}")
        if audio_ok:
            print("  → Proceed with Phase 1 (contact-mic primary signal)")
            if accel_ok:
                print("  → Accelerometer suitable for VAD fusion (Phase 2)")
            else:
                print("  → Use audio-only VAD (energy threshold); skip accel fusion")
        else:
            print("  → ❌ Contact-mic signal too weak. Check placement and try again.")
            print("     Ensure phone is pressed firmly, VOICE_RECOGNITION source is used,")
            print("     and device is not in a case that dampens contact vibration.")


if __name__ == "__main__":
    main()
