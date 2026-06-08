from pathlib import Path
import csv
import subprocess

# =====================
# CONFIG
# =====================

SRC_DATASET = Path("/content/datasets-1")
SRC_WAVS = SRC_DATASET / "wavs"

EVAL_CSV = SRC_DATASET / "metadata_eval.csv"

OUT_DIR = Path("/content/eval_44100")
OUT_WAVS = OUT_DIR / "wavs"
OUT_CSV = OUT_DIR / "metadata_eval.csv"

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


def convert_rows(rows):
    new_rows = []

    for i, row in enumerate(rows, 1):
        audio_file, text, speaker = row[0], row[1], row[2]

        wav_name = Path(audio_file).name
        src_wav = SRC_WAVS / wav_name
        dst_wav = OUT_WAVS / wav_name

        if not src_wav.exists():
            raise FileNotFoundError(f"Missing wav: {src_wav}")

        if not dst_wav.exists():
            convert_wav(src_wav, dst_wav)

        new_rows.append([f"wavs/{wav_name}", text, speaker])

        if i % 50 == 0:
            print(f"Converted eval {i}/{len(rows)}")

    return new_rows


# =====================
# MAIN
# =====================

rows = read_csv(EVAL_CSV)

print("Eval rows:", len(rows))
print("Output:", OUT_DIR)

new_rows = convert_rows(rows)
write_csv(OUT_CSV, new_rows)

print("DONE")
print("Eval wavs:", OUT_WAVS)
print("Eval csv :", OUT_CSV)