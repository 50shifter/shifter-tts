# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example: Save and load voice vectors (.pt files) for Qwen3-TTS.

This demonstrates how to:
1. Extract a voice vector from reference audio and save it to .pt
2. Load the saved voice vector and use it for TTS generation
   (without needing the original reference audio file)

Usage:
    python test_voice_vector_save_load.py \
        --ref-audio path/to/reference.wav \
        --ref-text "transcript of the reference audio" \
        --output-vector saved_voice.pt \
        --text "Text to synthesize with the cloned voice"
"""

import argparse
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser(
        description="Save and load voice vectors for Qwen3-TTS"
    )
    parser.add_argument(
        "--model-path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Path or HF repo ID of the Base model"
    )
    parser.add_argument(
        "--ref-audio", type=str, required=True,
        help="Path to reference audio file for voice extraction"
    )
    parser.add_argument(
        "--ref-text", type=str, default=None,
        help="Transcript of the reference audio (for ICL mode)"
    )
    parser.add_argument(
        "--output-vector", type=str, default="saved_voice.pt",
        help="Output path for the saved voice vector (.pt file)"
    )
    parser.add_argument(
        "--text", type=str, required=True,
        help="Text to synthesize using the loaded voice vector"
    )
    parser.add_argument(
        "--language", type=str, default="auto",
        help="Language for synthesis (auto, russian, english, chinese, etc.)"
    )
    parser.add_argument(
        "--x-vector-only", action="store_true",
        help="Use x_vector_only_mode: save/use only speaker embedding (ignore speech codes)"
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ===== Step 1: Load model =====
    print("\n=== Step 1: Loading model ===")
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    # ===== Step 2: Extract and save voice vector =====
    print(f"\n=== Step 2: Extracting voice vector from {args.ref_audio} ===")
    
    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only,
    )

    saved_path = tts.save_voice_vector(prompt_items, args.output_vector)
    print(f"[OK] Voice vector saved to: {saved_path}")

    # Show info about the saved vector
    item = prompt_items[0]
    if item.ref_code is not None:
        print(f"  ref_code shape: {item.ref_code.shape}")
    else:
        print("  ref_code: (not used, x_vector_only_mode=True)")
    print(f"  ref_spk_embedding dim: {item.ref_spk_embedding.shape[-1]}")
    print(f"  icl_mode: {item.icl_mode}")

    # ===== Step 3: Generate speech using the saved voice vector =====
    print(f"\n=== Step 3: Loading voice vector and generating speech ===")
    
    loaded_items = tts.load_voice_vector(saved_path)
    print(f"[OK] Voice vector loaded from: {saved_path}")

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()

    wavs, sr = tts.generate_voice_clone(
        text=args.text,
        language=args.language,
        voice_clone_prompt=loaded_items,
        x_vector_only_mode=args.x_vector_only,
    )

    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.time()

    duration = len(wavs[0]) / sr if sr > 0 else 0
    print(f"[OK] Generated speech: {duration:.2f}s @ {sr}Hz")
    print(f"[OK] Generation time: {t1 - t0:.3f}s")

    # Save output
    output_wav = "output_voice_vector.wav"
    sf.write(output_wav, wavs[0], sr)
    print(f"[OK] Audio saved to: {output_wav}")

    # ===== Step 4: Generate another utterance with the SAME voice vector =====
    print(f"\n=== Step 4: Generating another utterance (same voice, no audio needed!) ===")
    
    text2 = "This is a completely different sentence using the same cloned voice."
    if args.language.lower() == "russian":
        text2 = "Это совершенно другое предложение, использующее тот же клонированный голос."
    elif args.language.lower() == "chinese":
        text2 = "这是使用相同克隆语音的完全不同的句子。"

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()

    wavs2, sr2 = tts.generate_voice_clone(
        text=text2,
        language=args.language,
        voice_clone_prompt=loaded_items,  # Reusing the SAME loaded vector!
        x_vector_only_mode=args.x_vector_only,
    )

    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.time()

    duration2 = len(wavs2[0]) / sr2 if sr2 > 0 else 0
    print(f"[OK] Generated: {duration2:.2f}s @ {sr2}Hz")
    print(f"[OK] Generation time: {t1 - t0:.3f}s")

    output_wav2 = "output_voice_vector_2.wav"
    sf.write(output_wav2, wavs2[0], sr2)
    print(f"[OK] Audio saved to: {output_wav2}")

    print("\n=== Done! ===")
    print("You can now reuse the voice vector file (.pt) anytime without needing the original audio.")


if __name__ == "__main__":
    main()
