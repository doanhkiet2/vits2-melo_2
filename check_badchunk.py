# %%writefile /kaggle/working/vits2-melo/check_badchunk.py

# analyze_trunks_audio.py
import json
import math
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import pandas as pd

_book = "chuong_1_11"
_runlocal = False
if _runlocal:
    _runlocal = f"/mnt/f/{_book}"
else:
    _runlocal = ""

ROOT_DIR = Path(f"{_runlocal}/kaggle/working/infer_test/_segments")
OUTPUT_DIR = ROOT_DIR / "_analysis_tsv6"

TARGET_SR = 44100

# Nếu có list trunk lỗi thì điền tên file wav, mỗi dòng 1 tên.
# Không có thì để None.
BAD_LIST_TXT = None
# BAD_LIST_TXT = Path("/kaggle/working/bad_trunks.txt")


BANDS = [
    ("band_0_300", 0, 300),
    ("band_300_800", 300, 800),
    ("band_800_1500", 800, 1500),
    ("band_1500_3000", 1500, 3000),
    ("band_3000_6000", 3000, 6000),
    ("band_6000_12000", 6000, 12000),
    ("band_12000_20000", 12000, 20000),
]


def load_bad_set():
    if BAD_LIST_TXT is None or not BAD_LIST_TXT.exists():
        return set()

    return {
        line.strip()
        for line in BAD_LIST_TXT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def safe_mean(x):
    if len(x) == 0:
        return 0.0
    return float(np.mean(x))


def safe_std(x):
    if len(x) == 0:
        return 0.0
    return float(np.std(x))


def db(x, eps=1e-12):
    return 10.0 * np.log10(np.maximum(x, eps))


def band_energy_ratio(S_power, freqs, fmin, fmax, total_energy):
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return float(np.sum(S_power[mask, :]) / max(total_energy, 1e-12))


def read_text_for_wav(wav_path):
    txt_path = wav_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").replace("\n", " ").strip()
    return ""


def analyze_one(wav_path, bad_set):
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    if len(y) == 0:
        raise RuntimeError("empty audio")

    duration = len(y) / sr

    rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    crest = float(peak / max(rms, 1e-9))

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_power = S**2
    total_energy = float(np.sum(S_power) + 1e-12)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff85 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    rolloff95 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    bands = {}
    for name, fmin, fmax in BANDS:
        bands[name] = band_energy_ratio(S_power, freqs, fmin, fmax, total_energy)

    # Một vài ratio dễ soi lỗi "nghẹt / tối / kẹt e"
    low_mid = bands["band_300_800"] + bands["band_800_1500"]
    presence = bands["band_1500_3000"] + bands["band_3000_6000"]
    air = bands["band_6000_12000"] + bands["band_12000_20000"]

    text = read_text_for_wav(wav_path)

    row = {
        "label_bad": 1 if wav_path.name in bad_set or str(wav_path) in bad_set else 0,
        "file": str(wav_path),
        "name": wav_path.name,
        "duration_sec": duration,
        "text_len": len(text),
        "text": text,
        "rms_db": float(20 * np.log10(max(rms, 1e-12))),
        "peak": peak,
        "crest": crest,
        "rms_frame_mean_db": float(20 * np.log10(max(safe_mean(rms_frames), 1e-12))),
        "rms_frame_std": safe_std(rms_frames),
        "zcr_mean": safe_mean(zcr),
        "zcr_std": safe_std(zcr),
        "centroid_mean": safe_mean(centroid),
        "centroid_std": safe_std(centroid),
        "bandwidth_mean": safe_mean(bandwidth),
        "rolloff85_mean": safe_mean(rolloff85),
        "rolloff95_mean": safe_mean(rolloff95),
        "flatness_mean": safe_mean(flatness),
        "low_mid_ratio": low_mid,
        "presence_ratio": presence,
        "air_ratio": air,
        "presence_over_lowmid": presence / max(low_mid, 1e-12),
        "air_over_lowmid": air / max(low_mid, 1e-12),
        "air_over_presence": air / max(presence, 1e-12),
    }

    for name, value in bands.items():
        row[name] = value

    for i in range(13):
        row[f"mfcc{i+1}_mean"] = safe_mean(mfcc[i])
        row[f"mfcc{i+1}_std"] = safe_std(mfcc[i])

    return row


# =========================
# BAD TRUNK FILTER CONFIG
# =========================

WIDE_THRESHOLD = 18
NARROW_KEEP_PERCENT = 0.70
MIN_DENOM = 0.05
CAP_SCORE = 2.5

ALL_SIGN_RULES = {
    "rms_db": -1,
    "peak": -1,
    "crest": 1,
    "rms_frame_mean_db": -1,
    "rms_frame_std": -1,
    "zcr_mean": 1,
    "zcr_std": 1,
    "rolloff95_mean": 1,
    "flatness_mean": -1,
    "band_800_1500": -1,
    "band_12000_20000": 1,
    "mfcc1_std": -1,
    "mfcc2_std": -1,
    "mfcc3_mean": -1,
    "mfcc3_std": -1,
    "mfcc5_mean": -1,
    "mfcc6_mean": 1,
    "mfcc6_std": 1,
    "mfcc7_mean": 1,
    "mfcc7_std": 1,
    "mfcc8_mean": -1,
    "mfcc8_std": 1,
    "mfcc9_mean": -1,
    "mfcc9_std": -1,
    "mfcc10_mean": -1,
    "mfcc10_std": 1,
    "mfcc11_mean": 1,
    "mfcc11_std": -1,
    "mfcc12_std": 1,
    "mfcc13_mean": -1,
    "mfcc13_std": 1,
}

WIDE_RULES = [
    ("rms_db", -1, 1.0),
    ("peak", -1, 1.0),
    ("crest", 1, 1.0),
    ("rms_frame_mean_db", -1, 1.0),
    ("rms_frame_std", -1, 1.0),
    ("zcr_std", 1, 1.0),
    ("rolloff95_mean", 1, 1.0),
    ("flatness_mean", -1, 1.0),
    ("mfcc8_mean", -1, 1.0),
    ("mfcc10_mean", -1, 1.0),
    ("mfcc11_mean", 1, 1.0),
    ("mfcc13_mean", -1, 1.0),
    ("mfcc13_std", 1, 1.0),
    ("mfcc3_mean", -1, 0.5),
    ("band_800_1500", -1, 0.5),
    ("mfcc6_mean", 1, 0.5),
    ("mfcc12_std", 1, 0.5),
    ("mfcc7_mean", 1, 0.5),
    ("mfcc6_std", 1, 0.5),
    ("mfcc11_std", -1, 0.5),
    ("mfcc10_std", 1, 0.5),
    ("mfcc9_std", -1, 0.5),
    ("mfcc9_mean", -1, 0.5),
    ("mfcc5_mean", -1, 0.5),
    ("zcr_mean", 1, 0.5),
    ("mfcc2_std", -1, 0.5),
    ("mfcc7_std", 1, 0.5),
    ("mfcc8_std", 1, 0.5),
]


def percentile_inc(s, q):
    return float(np.nanpercentile(s.astype(float), q * 100))


def safe_baseline(series, sign, mode):
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return 1.0

    if mode == "wide":
        if sign == 1:
            return percentile_inc(s, 0.3)
        elif sign == -1:
            return percentile_inc(s, 0.7)
        else:
            return float(s.mean())

    return float(s.mean())


def calc_rel(series, baseline):
    denom = abs(baseline)
    if denom < 1e-12:
        denom = 1e-12
    return (pd.to_numeric(series, errors="coerce") - baseline) / denom


def calc_scale(rel, sign):
    r = pd.to_numeric(rel, errors="coerce").dropna()

    if sign == 1:
        side = r[r > 0]
    elif sign == -1:
        side = r[r < 0]
    else:
        side = r

    if len(side) == 0:
        return MIN_DENOM

    return max(abs(float(side.mean())), MIN_DENOM)


def score_one_rel(rel_value, sign, scale):
    if pd.isna(rel_value):
        return 0.0

    if sign == -1 and rel_value < 0:
        return min(abs(rel_value) / max(scale, MIN_DENOM), CAP_SCORE)

    if sign == 1 and rel_value > 0:
        return min(abs(rel_value) / max(scale, MIN_DENOM), CAP_SCORE)

    return 0.0


def filter_bad_trunks(df: pd.DataFrame, chapter_folder: str):
    df = df.copy()

    # ---------- STEP 1: wide score ----------
    df["wide_score"] = 0.0

    for metric, sign, weight in WIDE_RULES:
        if metric not in df.columns:
            print(f"[WARN] missing metric: {metric}")
            continue

        baseline = safe_baseline(df[metric], sign, mode="wide")
        rel = calc_rel(df[metric], baseline)
        scale = calc_scale(rel, sign)

        df[f"wide_rel_{metric}"] = rel
        df[f"wide_part_{metric}"] = (
            rel.apply(lambda x: score_one_rel(x, sign, scale)) * weight
        )

        df["wide_score"] += df[f"wide_part_{metric}"]

    df["wide_pass"] = df["wide_score"] > WIDE_THRESHOLD

    # ---------- STEP 2: narrow count ----------
    df["narrow_count"] = 0

    for metric, sign in ALL_SIGN_RULES.items():
        if sign == 0 or metric not in df.columns:
            continue

        baseline = safe_baseline(df[metric], sign, mode="narrow")
        rel = calc_rel(df[metric], baseline)

        df[f"narrow_rel_{metric}"] = rel
        df[f"narrow_hit_{metric}"] = ((rel * sign) > 0).astype(int)
        df["narrow_count"] += df[f"narrow_hit_{metric}"]

    wide_df = df[df["wide_pass"]].copy()

    if len(wide_df) == 0:
        df["final_bad"] = False
        return [], df

    # Excel: PERCENTILE.INC(IF(wide_pass, narrow_count), 0.3)
    narrow_cutoff = percentile_inc(wide_df["narrow_count"], 0.3)

    df["final_bad"] = df["wide_pass"] & (df["narrow_count"] >= narrow_cutoff)

    bad_df = df[df["final_bad"]].copy()

    bad_items = []

    for _, row in bad_df.iterrows():
        bad_items.append(
            {
                "chapter_folder": chapter_folder,
                "name": row.get("name", ""),
                "file": row.get("file", ""),
                "text": row.get("text", ""),
                "wide_score": float(row.get("wide_score", 0.0)),
                "narrow_count": int(row.get("narrow_count", 0)),
            }
        )

    return bad_items, df


def analyze_folder(chapter_dir: Path, bad_set):
    wavs = sorted(chapter_dir.glob("*.wav"))

    if not wavs:
        print(f"[SKIP] Không có wav: {chapter_dir}")
        return

    rows = []

    for i, wav_path in enumerate(wavs, start=1):
        try:
            row = analyze_one(wav_path, bad_set)
            row["chapter_folder"] = chapter_dir.name
            rows.append(row)
            print(f"[OK] {chapter_dir.name} {i}/{len(wavs)} {wav_path.name}")
        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")

    if not rows:
        print(f"[SKIP] Không phân tích được file nào: {chapter_dir}")
        return

    keys = list(rows[0].keys())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    bad_items, df = filter_bad_trunks(df, chapter_dir.name)

    bad_json = OUTPUT_DIR / f"{chapter_dir.name}_bad_trunks.json"
    with bad_json.open("w", encoding="utf-8") as f:
        json.dump(bad_items, f, ensure_ascii=False, indent=2)

    print(f"[BAD] {chapter_dir.name}: {len(bad_items)}")
    print("[DONE BAD JSON]", bad_json)

    out_xlsx = OUTPUT_DIR / f"{chapter_dir.name}.xlsx"

    df.to_excel(
        out_xlsx,
        index=False,
        engine="openpyxl",
    )

    print("[DONE]", out_xlsx)


def main():
    bad_set = load_bad_set()

    chapter_dirs = sorted(
        p for p in ROOT_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")
    )

    if not chapter_dirs:
        raise RuntimeError(f"Không tìm thấy folder chương trong {ROOT_DIR}")

    print(f"FOUND {len(chapter_dirs)} chapter folders")

    for chapter_dir in chapter_dirs:
        print("\n==============================")
        print("ANALYZE:", chapter_dir.name)
        print("==============================")
        analyze_folder(chapter_dir, bad_set)

    print("\nALL DONE:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
