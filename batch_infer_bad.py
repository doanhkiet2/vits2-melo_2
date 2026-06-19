# %%writefile /kaggle/working/vits2-melo/batch_infer_bad.py

import os
import json
import argparse
from pathlib import Path

import torch

from melo.api import TTS


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bad_json", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--language", default="VI")
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--config_path", default=None)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--use_hf", action="store_true")

    parser.add_argument("--sdp_ratio", type=float, default=0.2)
    parser.add_argument("--noise_scale", type=float, default=0.6)
    parser.add_argument("--noise_scale_w", type=float, default=0.8)
    parser.add_argument("--speed", type=float, default=1.0)

    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    bad_json = Path(args.bad_json)
    output_dir = Path(args.output_dir)

    if not bad_json.exists():
        raise FileNotFoundError(f"bad_json not found: {bad_json}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("========== BAD TRUNK INFER ==========")
    print("bad_json   :", bad_json)
    print("output_dir :", output_dir)
    print("ckpt_path  :", args.ckpt_path)
    print("config_path:", args.config_path)
    print("device     :", args.device)
    print("=====================================")

    tts = TTS(
        language=args.language,
        device=args.device,
        use_hf=args.use_hf,
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
    )

    manifest = tts.bad_trunks_to_files(
        bad_json_path=str(bad_json),
        output_dir=str(output_dir),
        speaker_id=args.speaker_id,
        sdp_ratio=args.sdp_ratio,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        speed=args.speed,
        quiet=args.quiet,
    )

    print("DONE")
    print("generated:", len(manifest))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
