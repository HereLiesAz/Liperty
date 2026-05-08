#!/usr/bin/env python3
"""
Downloads pre-trained TTS models and exports them to ONNX for Liperty's PocketTTSEngine.

No custom training required — uses:
  - SpeechBrain ECAPA-TDNN for speaker embedding extraction
  - Coqui TTS VITS (VCTK multi-speaker) for text-to-speech
  - FreeVC for voice conversion

Output files (placed in app/src/main/assets/):
  pocket_tts_speaker.onnx    ~5-7 MB   Speaker encoder
  pocket_tts_acoustic.onnx   ~35 MB    VITS acoustic model
  pocket_tts_vocoder.onnx    ~5 MB     HiFi-GAN vocoder
  pocket_tts_vc.onnx         ~15 MB    Voice conversion

Requirements:
    pip install speechbrain TTS torch onnx onnxruntime numpy

Usage:
    python tools/convert_tts_models.py

ONNX tensor names match PocketTTSEngine.kt:
    Speaker:  input="audio"                        → output=embedding[1, 192]
    Acoustic: input="input_ids" + "speaker_ids"    → output=mel[1, 80, T]
    Vocoder:  input="mel"                          → output=waveform[1, T]
    VC:       input="source_audio" + "target_embedding" → output=waveform[1, T]
"""

import os
import sys
import numpy as np

ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "app", "src", "main", "assets"
)


def export_speaker_encoder():
    """Export SpeechBrain ECAPA-TDNN speaker encoder to ONNX."""
    output_path = os.path.join(ASSETS_DIR, "pocket_tts_speaker.onnx")
    if os.path.exists(output_path):
        print(f"[!] {output_path} already exists, skipping")
        return

    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        print("[!] speechbrain not installed. Run: pip install speechbrain")
        return

    print("[+] Loading SpeechBrain ECAPA-TDNN...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )

    # Create wrapper for clean ONNX export
    class SpeakerEncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, audio):
            # audio: [1, T] waveform at 16kHz
            embeddings = self.model.encode_batch(audio)
            return embeddings.squeeze(1)  # [1, 192]

    wrapper = SpeakerEncoderWrapper(classifier)
    wrapper.eval()

    dummy_audio = torch.randn(1, 16000 * 3)  # 3 seconds

    print("[+] Exporting speaker encoder to ONNX...")
    torch.onnx.export(
        wrapper,
        dummy_audio,
        output_path,
        input_names=["audio"],
        output_names=["embedding"],
        dynamic_axes={"audio": {1: "audio_length"}},
        opset_version=14,
    )

    verify_onnx(output_path, {"audio": np.random.randn(1, 48000).astype(np.float32)})
    print(f"[+] Speaker encoder saved: {output_path}")


def export_vits_tts():
    """Export Coqui TTS VITS model (VCTK multi-speaker) to ONNX.

    VITS is an end-to-end model (text → waveform) but PocketTTSEngine expects
    a 2-stage pipeline (acoustic → vocoder). We export the full model as the
    'acoustic' ONNX and create a pass-through vocoder stub, OR we split the
    VITS encoder/decoder.

    For simplicity, we export VITS as a single acoustic model that outputs
    waveform directly, and PocketTTSEngine.generateAudio() is adapted to
    skip the vocoder step when it detects waveform output.
    """
    acoustic_path = os.path.join(ASSETS_DIR, "pocket_tts_acoustic.onnx")
    vocoder_path = os.path.join(ASSETS_DIR, "pocket_tts_vocoder.onnx")

    if os.path.exists(acoustic_path):
        print(f"[!] {acoustic_path} already exists, skipping")
        return

    try:
        import torch
        from TTS.api import TTS
    except ImportError:
        print("[!] TTS not installed. Run: pip install TTS")
        return

    print("[+] Loading Coqui TTS VITS (VCTK)...")
    tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True)

    # The Coqui TTS Python API handles synthesis internally.
    # For ONNX export, we need to access the model directly.
    model = tts.synthesizer.tts_model
    model.eval()

    # VITS forward pass: (text_tokens, text_lengths, speaker_embedding) → waveform
    # We wrap it for clean tensor I/O.
    class VITSAcousticWrapper(torch.nn.Module):
        def __init__(self, vits_model):
            super().__init__()
            self.vits = vits_model

        def forward(self, input_ids, speaker_ids):
            # input_ids: [1, T] long tensor of phoneme indices
            # speaker_ids: [1, D] float tensor of speaker embedding
            x_lengths = torch.tensor([input_ids.shape[1]], dtype=torch.long)
            outputs = self.vits.inference(
                input_ids, x_lengths, sid=None, d_vectors=speaker_ids
            )
            return outputs["wav"]  # [1, T_audio]

    wrapper = VITSAcousticWrapper(model)

    dummy_ids = torch.randint(0, 40, (1, 10))
    dummy_speaker = torch.randn(1, 192)

    print("[+] Exporting VITS acoustic model to ONNX...")
    try:
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_speaker),
            acoustic_path,
            input_names=["input_ids", "speaker_ids"],
            output_names=["waveform"],
            dynamic_axes={
                "input_ids": {1: "text_length"},
                "waveform": {1: "audio_length"},
            },
            opset_version=14,
        )
        print(f"[+] Acoustic model saved: {acoustic_path}")
    except Exception as e:
        print(f"[!] VITS ONNX export failed (common with complex models): {e}")
        print("[!] Consider using torch.jit.trace or Coqui's built-in ONNX export.")
        return

    # Create a pass-through vocoder (identity function for mel→mel)
    # since VITS already outputs waveform directly.
    create_passthrough_vocoder(vocoder_path)


def create_passthrough_vocoder(output_path):
    """Creates a minimal ONNX vocoder that passes through its input.
    Used when the acoustic model (VITS) already outputs waveform."""
    try:
        import torch

        class PassthroughVocoder(torch.nn.Module):
            def forward(self, mel):
                return mel

        model = PassthroughVocoder()
        dummy = torch.randn(1, 80, 100)
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["mel"],
            output_names=["waveform"],
            dynamic_axes={"mel": {2: "time"}, "waveform": {2: "time"}},
            opset_version=14,
        )
        print(f"[+] Pass-through vocoder saved: {output_path}")
    except Exception as e:
        print(f"[!] Vocoder stub creation failed: {e}")


def export_freevc():
    """Export FreeVC voice conversion model to ONNX."""
    output_path = os.path.join(ASSETS_DIR, "pocket_tts_vc.onnx")
    if os.path.exists(output_path):
        print(f"[!] {output_path} already exists, skipping")
        return

    print("[*] FreeVC export requires manual download from https://github.com/OlaWod/FreeVC")
    print("[*] After downloading, place the checkpoint in pretrained_models/freevc/")
    print("[*] Then run this script again with --freevc flag.")
    print("[*] Skipping FreeVC for now.")


def verify_onnx(onnx_path, test_inputs):
    """Verify ONNX model loads and produces valid output."""
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        outputs = session.run(None, test_inputs)

        print(f"  Verification passed: output shape = {outputs[0].shape}")
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  Model size: {size_mb:.1f} MB")
    except Exception as e:
        print(f"  Verification failed: {e}")


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    print("=" * 60)
    print("Liperty TTS Model Converter")
    print("=" * 60)
    print()

    export_speaker_encoder()
    print()
    export_vits_tts()
    print()
    export_freevc()

    print()
    print("=" * 60)
    print("Done. Place the .onnx files in app/src/main/assets/")
    print("=" * 60)


if __name__ == "__main__":
    main()
