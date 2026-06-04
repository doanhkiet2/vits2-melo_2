from pathlib import Path
import csv

SRC_DATASET = Path("/content/datasets-1")
TRAIN_CSV = SRC_DATASET / "metadata_train.csv"
EVAL_CSV = SRC_DATASET / "metadata_eval.csv"

OUT_ROOT = Path("/content/vits-melo/melo/data/vi_speaker_full")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    text = text.strip()
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("|", " ")
    text = " ".join(text.split())

    if text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3].strip()
    elif text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    return text

def convert_csv(csv_path, out_list):
    lines = []
    missing = 0
    bad = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|", quotechar='"')

        for row in reader:
            if len(row) < 3:
                bad += 1
                continue

            audio_rel, text, speaker = row[0], row[1], row[2]

            if audio_rel in ["audio_file", "audio", "path"]:
                continue

            wav = SRC_DATASET / audio_rel

            if not wav.exists():
                print("MISSING:", wav)
                missing += 1
                continue

            text = clean_text(text)

            if not text:
                bad += 1
                continue

            lines.append(f"{wav}|vi_speaker|VI|{text}")

    out_list.write_text("\n".join(lines), encoding="utf-8")
    print(out_list, "rows:", len(lines), "missing:", missing, "bad:", bad)

convert_csv(TRAIN_CSV, OUT_ROOT / "train_raw.list")
convert_csv(EVAL_CSV, OUT_ROOT / "val_raw.list")

metadata = OUT_ROOT / "metadata.list"
metadata.write_text(
    (OUT_ROOT / "train_raw.list").read_text(encoding="utf-8").strip()
    + "\n"
    + (OUT_ROOT / "val_raw.list").read_text(encoding="utf-8").strip(),
    encoding="utf-8"
)

print("metadata rows:", len(metadata.read_text(encoding="utf-8").splitlines()))