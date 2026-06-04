from pathlib import Path
import csv
import shutil
import subprocess
import math

# =====================
# CONFIG
# =====================

SRC_DATASET = Path("/content/datasets-1")
SRC_WAVS = SRC_DATASET / "wavs"

TRAIN_CSV = SRC_DATASET / "metadata_train.csv"
EVAL_CSV = SRC_DATASET / "metadata_eval.csv"

OUT_ROOT = Path("/content/datasets_44100_split")

PART_NO = 1   # đổi thành 2 hoặc 3 khi train đợt sau
NUM_PARTS = 3
TARGET_SR = 44100

# =====================
# UTILS
# =====================

def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 3:
                continue

            # bỏ header
            if row[0].strip() == "audio_file":
                continue

            rows.append(row)

    return rows

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text", "speaker_name"])
        writer.writerows(rows)


def convert_wav(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)

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


def convert_rows(rows, out_wavs):
    new_rows = []

    for i, row in enumerate(rows, 1):
        audio_file, text, speaker = row[0], row[1], row[2]

        wav_name = Path(audio_file).name
        src_wav = SRC_WAVS / wav_name
        dst_wav = out_wavs / wav_name

        if not src_wav.exists():
            raise FileNotFoundError(f"Missing wav: {src_wav}")

        if not dst_wav.exists():
            convert_wav(src_wav, dst_wav)

        new_rows.append([f"wavs/{wav_name}", text, speaker])

        if i % 100 == 0:
            print(f"Converted {i}/{len(rows)}")

    return new_rows


# =====================
# MAIN
# =====================

train_rows = read_csv(TRAIN_CSV)
eval_rows = read_csv(EVAL_CSV)

print("Train rows:", len(train_rows))
print("Eval rows :", len(eval_rows))

part_size = math.ceil(len(train_rows) / NUM_PARTS)

part_idx = PART_NO - 1
start = part_idx * part_size
end = min(start + part_size, len(train_rows))

part_train_rows = train_rows[start:end]

out_dir = OUT_ROOT / f"part_{PART_NO}"
out_wavs = out_dir / "wavs"

print(f"========== PART {PART_NO} ==========")
print(f"Train rows: {len(part_train_rows)}")
print(f"Output: {out_dir}")

new_train = convert_rows(part_train_rows, out_wavs)
new_eval = convert_rows(eval_rows, out_wavs)

write_csv(out_dir / "metadata_train.csv", new_train)
write_csv(out_dir / "metadata_eval.csv", new_eval)

print("DONE")