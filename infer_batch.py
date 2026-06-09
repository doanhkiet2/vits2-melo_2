# infer_batch.py

import os
import sys
import shutil
import subprocess
from pathlib import Path


MODEL = "/content/vits2-melo/logs/vi_speaker_merged/G_108000.pth"
OUT_DIR = Path("/content/infer_test")
MELO_DIR = "/content/vits2-melo"
INFER_PY = "/content/vits2-melo/melo/infer.py"

TEXTS = [
    "Hắn chậm rãi mở mắt, nhìn về phía chân trời xa xăm.",
    "Nàng khẽ thở dài, trong lòng dâng lên một cảm giác bất an khó tả.",
    "Trên con đường nhỏ phủ đầy sương sớm, tiếng bước chân vang lên rất nhẹ.",
    "Tiểu Thanh khẽ mím môi, ánh mắt vừa có chút do dự vừa mang theo vẻ kiên định hiếm thấy.",
    "Ba trăm sáu mươi lăm ngày đã trôi qua, nhưng ký ức về đêm mưa năm ấy vẫn còn rõ ràng như mới hôm qua.",
    "Nếu không tận mắt chứng kiến, e rằng chẳng ai tin nổi chuyện kỳ lạ như vậy lại thực sự xảy ra.",
    "Lâm Tinh Thừa đứng trên vách núi cheo leo, lặng lẽ nhìn những đám mây cuồn cuộn phía cuối chân trời.",
    "Từng tia chớp xé toạc bầu trời đen kịt, khiến đám trẻ hoảng hốt nép sát vào nhau mà không dám lên tiếng.",
    "Mặc dù đã chuẩn bị tâm lý từ trước, hắn vẫn không khỏi kinh ngạc khi phát hiện bí mật được che giấu suốt hơn hai mươi năm.",
    "Trong khoảnh khắc sinh tử ấy, mọi do dự, sợ hãi và hối tiếc dường như đều tan biến, chỉ còn lại một ý chí mãnh liệt muốn tiếp tục sống.",
]


def list_wavs():
    return set(OUT_DIR.rglob("*.wav"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Python:", sys.executable)
    print("Model :", MODEL)
    print("Out   :", OUT_DIR)
    print()

    for idx, text in enumerate(TEXTS, start=1):
        final_out = OUT_DIR / f"{idx:04d}.wav"

        before = list_wavs()

        cmd = [
            sys.executable,
            INFER_PY,
            "-m", MODEL,
            "-l", "VI",
            "-o", str(OUT_DIR),
            "-t", text,
        ]

        print(f"[{idx:02d}/{len(TEXTS)}] {text}")

        result = subprocess.run(
            cmd,
            env={
                **os.environ,
                "PYTHONPATH": MELO_DIR,
            },
            text=True,
            capture_output=True,
        )

        if result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"Infer failed at item {idx}")

        after = list_wavs()
        new_files = list(after - before)

        if new_files:
            newest = max(new_files, key=lambda p: p.stat().st_mtime)
        else:
            wavs = list(after)
            if not wavs:
                raise RuntimeError("Không tìm thấy file wav nào sau khi infer.")
            newest = max(wavs, key=lambda p: p.stat().st_mtime)

        if newest != final_out:
            if final_out.exists():
                final_out.unlink()
            shutil.move(str(newest), str(final_out))

        print("Saved:", final_out)
        print()

    print("DONE")


if __name__ == "__main__":
    main()