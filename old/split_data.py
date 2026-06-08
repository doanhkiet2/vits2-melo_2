from pathlib import Path
import subprocess
import math

# =====================
# CONFIG
# =====================

PART_NO = 1          # đổi 1 / 2 / 3
NUM_PARTS = 3
TARGET_SR = 44100

TRAIN_LIST = Path("/content/vits2-melo/melo/data/vi_speaker_full/train.list")
VAL_LIST = Path("/content/vits2-melo/melo/data/vi_speaker_full/val.list")

OUT_DIR = Path("/content/vits2-melo/melo/data/datasets_44100_split/vi_speaker_44100")
OUT_WAVS = OUT_DIR / "wavs"

OUT_TRAIN_LIST = OUT_DIR / "train.list"
OUT_VAL_LIST = OUT_DIR / "val.list"

# =====================
# UTILS
# =====================

def read_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def write_list(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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


def process_lines(lines, out_list_path, label):
    new_lines = []

    for i, line in enumerate(lines, 1):
        cols = line.split("|")

        if len(cols) < 2:
            raise ValueError(f"Bad line: {line}")

        src_audio = Path(cols[0])
        wav_name = src_audio.name
        dst_audio = OUT_WAVS / wav_name

        if not src_audio.exists():
            raise FileNotFoundError(f"Missing source wav: {src_audio}")

        if not dst_audio.exists():
            convert_wav(src_audio, dst_audio)

        cols[0] = str(dst_audio)
        new_lines.append("|".join(cols))

        if i % 100 == 0:
            print(f"{label}: {i}/{len(lines)}")

    write_list(out_list_path, new_lines)


# =====================
# MAIN
# =====================

train_lines = read_list(TRAIN_LIST)
val_lines = read_list(VAL_LIST)

print("Source train:", len(train_lines))
print("Source val  :", len(val_lines))

part_size = math.ceil(len(train_lines) / NUM_PARTS)

part_idx = PART_NO - 1
start = part_idx * part_size
end = min(start + part_size, len(train_lines))

part_train_lines = train_lines[start:end]

print()
print(f"========== PART {PART_NO}/{NUM_PARTS} ==========")
print("Start index:", start)
print("End index  :", end)
print("Train rows :", len(part_train_lines))
print("Val rows   :", len(val_lines))
print("Output     :", OUT_DIR)

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_WAVS.mkdir(parents=True, exist_ok=True)

process_lines(part_train_lines, OUT_TRAIN_LIST, "train")
process_lines(val_lines, OUT_VAL_LIST, "val")

print()
print("DONE")
print("Train list:", OUT_TRAIN_LIST)
print("Val list  :", OUT_VAL_LIST)
print("Wavs dir  :", OUT_WAVS)