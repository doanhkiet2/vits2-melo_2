# filter_bad_source_train_list.py

import json
from pathlib import Path
from collections import Counter

# =====================
# CONFIG
# =====================

TRAIN_IN = Path("/content/vits2-melo/xuly/val2.list")
TRAIN_OUT = Path("/content/vits2-melo/xuly/val3.list")
REMOVED_OUT = Path("/content/vits2-melo/xuly/val.removed_bad_source.list")

LOUDNESS_REPORT = Path("/content/vits2-melo/xuly/source_cloudness_report.jsonl")
AUDIO_MAP = Path("/content/vits2-melo/xuly/audio_name_map.json")

BAD_STATUS = "BAD_SOURCE"


def load_bad_prefixes(report_path: Path):
    bad_prefixes = set()

    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            if obj.get("loudness_status") != BAD_STATUS:
                continue

            filename = obj.get("filename") or Path(obj.get("path", "")).name

            # "_x6Kav8s5jSo.wav" -> "_x6Kav8s5jSo"
            prefix = Path(filename).stem
            bad_prefixes.add(prefix)

    return bad_prefixes


def load_bad_audio_names(map_path: Path, bad_prefixes: set):
    data = json.loads(map_path.read_text(encoding="utf-8"))

    bad_audio_names = set()
    prefix_to_audio_name = {}

    for item in data["items"]:
        audio_name = item["audio_name"]
        prefix = item["prefix"]

        if prefix in bad_prefixes:
            bad_audio_names.add(audio_name)
            prefix_to_audio_name[prefix] = audio_name

    return bad_audio_names, prefix_to_audio_name


def train_line_to_audio_name(line: str):
    wav_path = line.split("|", 1)[0]
    wav_name = Path(wav_path).name

    # "hdn - Copy (5)__final_0001897.wav"
    # -> "hdn - Copy (5)"
    if "__final_" in wav_name:
        return wav_name.split("__final_", 1)[0]

    return None


def main():
    bad_prefixes = load_bad_prefixes(LOUDNESS_REPORT)
    bad_audio_names, prefix_to_audio_name = load_bad_audio_names(AUDIO_MAP, bad_prefixes)

    print("Bad prefixes:", len(bad_prefixes))
    for p in sorted(bad_prefixes):
        print("  BAD PREFIX:", p, "=>", prefix_to_audio_name.get(p, "NOT_FOUND_IN_MAP"))

    print()
    print("Bad audio_names:", len(bad_audio_names))
    for name in sorted(bad_audio_names):
        print("  BAD AUDIO_NAME:", name)

    total = 0
    kept = 0
    removed = 0
    removed_counter = Counter()

    with TRAIN_IN.open("r", encoding="utf-8") as fin, \
         TRAIN_OUT.open("w", encoding="utf-8") as fout, \
         REMOVED_OUT.open("w", encoding="utf-8") as frem:

        for line in fin:
            if not line.strip():
                continue

            total += 1
            audio_name = train_line_to_audio_name(line)

            if audio_name in bad_audio_names:
                removed += 1
                removed_counter[audio_name] += 1
                frem.write(line)
            else:
                kept += 1
                fout.write(line)

    print()
    print("DONE")
    print("Input  :", TRAIN_IN)
    print("Output :", TRAIN_OUT)
    print("Removed:", REMOVED_OUT)
    print("Total  :", total)
    print("Kept   :", kept)
    print("Removed:", removed)

    print()
    print("Removed by audio_name:")
    for name, count in removed_counter.most_common():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()