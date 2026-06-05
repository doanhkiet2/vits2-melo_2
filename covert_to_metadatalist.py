from pathlib import Path
import csv
import subprocess
from tqdm import tqdm

# =====================
# CONFIG
# =====================

PART_NO = 1  # đổi 1 / 2 / 3
TARGET_SR = 44100

SRC_DATASET = Path("/content/datasets-1")
SRC_WAVS = SRC_DATASET / "wavs"

INPUT_CSV = SRC_DATASET / f"metadata_train_part{PART_NO}.csv"

OUT_DIR = Path(f"/content/vits2-melo/melo/data/vi_speaker_44100_part{PART_NO}")
OUT_WAVS = OUT_DIR / "wavs"
OUT_METADATA = OUT_DIR / "metadata.list"

SPEAKER = "VI-default"
LANGUAGE = "VI"

# =====================
# UTILS
# =====================

def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if not row:
                continue
            if row[0].strip() == "audio_file":
                continue
            if len(row) < 2:
                continue
            rows.append(row)
    return rows


def resolve_audio_path(audio_value):
    p = Path(audio_value)

    if p.is_absolute():
        return p

    # CSV thường có dạng wavs/xxx.wav
    p1 = SRC_DATASET / p
    if p1.exists():
        return p1

    # fallback: chỉ lấy tên file trong /content/datasets-1/wavs
    p2 = SRC_WAVS / p.name
    return p2


def convert_wav(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(src),
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-sample_fmt", "s16",
        str(dst),
    ]

    subprocess.run(cmd, check=True)


# =====================
# MAIN
# =====================

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_WAVS.mkdir(parents=True, exist_ok=True)

rows = read_csv(INPUT_CSV)

print("Input CSV :", INPUT_CSV)
print("Rows      :", len(rows))
print("Output dir:", OUT_DIR)

lines = []
missing = 0

for row in tqdm(rows):
    audio_value = row[0]
    text = row[1].strip()

    src_audio = resolve_audio_path(audio_value)
    if not src_audio.exists():
        missing += 1
        print("MISSING:", src_audio)
        continue

    dst_audio = OUT_WAVS / src_audio.name
    convert_wav(src_audio, dst_audio)

    # format cho preprocess gốc:
    # utt|spk|language|text
    lines.append(f"{dst_audio}|{SPEAKER}|{LANGUAGE}|{text}")

with open(OUT_METADATA, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print()
print("DONE")
print("Written:", len(lines))
print("Missing:", missing)
print("Metadata:", OUT_METADATA)
print("Wavs    :", OUT_WAVS)