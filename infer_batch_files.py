# %%writefile /kaggle/working/vits2-melo/infer_batch_files.py
# infer_batch_files_2gpu.py

import os
import json
import shutil
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import contextlib
import warnings

QUIET_INTERNAL = True

MODEL = "/kaggle/working/vits2-melo/logs/vi_speaker_merged/G_104000.pth"
MELO_DIR = "/kaggle/working/vits2-melo"

INPUT_DIR = Path("/kaggle/working/novel_full_clean")
OUT_DIR = Path("/kaggle/working/infer_test")

FILELIST_PATH = OUT_DIR / "filelist.jsonl"
LOG_PATH = OUT_DIR / "generate_log.jsonl"
CONFIG = "/kaggle/working/vits2-melo/logs/vi_speaker_merged/config.json"
START_INDEX = 1
END_INDEX = None

GPU_IDS = [0, 1]
LANGUAGE = "VI"
SPEAKER_ID = 0

SPEED = 1.0
SDP_RATIO = 0.2
NOISE_SCALE = 0.6
NOISE_SCALE_W = 0.8

SKIP_EXISTING = True


@contextlib.contextmanager
def suppress_internal_output():
    if not QUIET_INTERNAL:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_log(data):
    data = {"time": now(), **data}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def build_filelist_if_missing():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if FILELIST_PATH.exists():
        print("Use existing filelist:", FILELIST_PATH)
        return

    txt_files = sorted(INPUT_DIR.rglob("*.txt"))

    if not txt_files:
        raise RuntimeError(f"Không tìm thấy file .txt trong {INPUT_DIR}")

    with FILELIST_PATH.open("w", encoding="utf-8") as f:
        for idx, txt_path in enumerate(txt_files, start=1):
            out_path = OUT_DIR / f"{idx:05d}_{txt_path.stem}.wav"
            row = {
                "index": idx,
                "text_path": str(txt_path),
                "output_path": str(out_path),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Created filelist:", FILELIST_PATH)


def load_filelist():
    rows = []

    with FILELIST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            idx = row["index"]

            if idx < START_INDEX:
                continue

            if END_INDEX is not None and idx > END_INDEX:
                continue

            rows.append(row)

    return rows


def split_rows_for_workers(rows, num_workers):
    buckets = [[] for _ in range(num_workers)]

    for i, row in enumerate(rows):
        buckets[i % num_workers].append(row)

    return buckets


def worker_main(gpu_id, rows):
    os.environ["PYTHONPATH"] = MELO_DIR
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import sys
    import torch
    import soundfile as sf

    sys.path.insert(0, MELO_DIR)

    from melo.api import TTS

    print(f"[GPU {gpu_id}] Loading model...")

    with suppress_internal_output():
        tts = TTS(
            language=LANGUAGE,
            device="cuda",
            use_hf=False,
            config_path=CONFIG,
            ckpt_path=MODEL,
        )

    print(f"[GPU {gpu_id}] Loaded. Jobs:", len(rows))

    for row in rows:
        idx = row["index"]
        text_path = Path(row["text_path"])
        output_path = Path(row["output_path"])

        try:
            if (
                SKIP_EXISTING
                and output_path.exists()
                and output_path.stat().st_size > 0
            ):
                print(f"[GPU {gpu_id}] [SKIP] {idx:05d}")
                safe_log(
                    {
                        "index": idx,
                        "gpu": gpu_id,
                        "status": "SKIP_EXISTS",
                        "text_path": str(text_path),
                        "output_path": str(output_path),
                    }
                )
                continue

            text = text_path.read_text(encoding="utf-8").strip()

            if not text:
                print(f"[GPU {gpu_id}] [SKIP_EMPTY] {idx:05d}")
                safe_log(
                    {
                        "index": idx,
                        "gpu": gpu_id,
                        "status": "SKIP_EMPTY_TEXT",
                        "text_path": str(text_path),
                        "output_path": str(output_path),
                    }
                )
                continue

            print(f"[GPU {gpu_id}] [RUN] {idx:05d} {text_path.name}")

            safe_log(
                {
                    "index": idx,
                    "gpu": gpu_id,
                    "status": "START",
                    "text_path": str(text_path),
                    "output_path": str(output_path),
                    "text_chars": len(text),
                }
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with suppress_internal_output():
                audio = tts.tts_to_file(
                    text=text,
                    speaker_id=SPEAKER_ID,
                    output_path=None,
                    sdp_ratio=SDP_RATIO,
                    noise_scale=NOISE_SCALE,
                    noise_scale_w=NOISE_SCALE_W,
                    speed=SPEED,
                    quiet=True,
                )

            tmp_path = output_path.with_suffix(".tmp.wav")
            sf.write(tmp_path, audio, tts.hps.data.sampling_rate)

            if output_path.exists():
                output_path.unlink()

            shutil.move(str(tmp_path), str(output_path))

            safe_log(
                {
                    "index": idx,
                    "gpu": gpu_id,
                    "status": "DONE",
                    "text_path": str(text_path),
                    "output_path": str(output_path),
                    "wav_size": output_path.stat().st_size,
                }
            )

            print(f"[GPU {gpu_id}] [DONE] {idx:05d}")

            torch.cuda.empty_cache()

        except Exception as e:
            safe_log(
                {
                    "index": idx,
                    "gpu": gpu_id,
                    "status": "ERROR",
                    "text_path": str(text_path),
                    "output_path": str(output_path),
                    "error": repr(e),
                }
            )
            print(f"[GPU {gpu_id}] [ERROR] {idx:05d}: {e}")
            raise


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    build_filelist_if_missing()
    rows = load_filelist()

    print("Total jobs:", len(rows))
    print("GPU ids:", GPU_IDS)
    print("Start:", START_INDEX)
    print("End:", END_INDEX)
    print("Out:", OUT_DIR)

    buckets = split_rows_for_workers(rows, len(GPU_IDS))

    processes = []

    for gpu_id, bucket in zip(GPU_IDS, buckets):
        p = mp.Process(target=worker_main, args=(gpu_id, bucket))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for p in processes:
        if p.exitcode != 0:
            raise RuntimeError(f"Worker failed with exit code {p.exitcode}")

    print("ALL DONE")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
