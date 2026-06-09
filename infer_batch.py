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

    "Nàng khẽ thở dài, trong lòng dâng lên một cảm giác bất an khó tả.",
    "như hoa mẫu đơn nở rộ trên cành vào cuối xuân, khó tả xiết bao, hô hấp không khỏi nghẹn lại.",
    "Ban đầu có sợ, nhưng lúc này nhìn thấy Lư Đại Lang như vậy, thấy Lư Đại Lang nhốt nàng ta thế nào, làm sao xử lý nàng ta, trong lòng đột nhiên dâng lên một loại cảm giác sảng khoái khó tả.",
    "hoặc là trong ánh mắt của Thẩm Liệt nhìn thấy bóng dáng của mình, hoặc là ánh mắt của thiếu niên vừa chuyên chú lại có chút dịu dàng khó tả, trái tim của Tăng La đột nhiên nhảy lên, có chút kỳ lạ.",
    "Ngươi nói chuyện nhẹ nhàng từ tốn. toát ra sự dịu dàng từ trong xương, nhưng lại dường như mang theo một nỗi u sầu khó tả.",
    "không thể diễn tả thành lời, chỉ có lồng ngực đập thình thịch đến trướng lên, tất cả đều là niềm vui sướng khó nói khó tả lại khiến người yêu muốn chết.",
    "Vốn dĩ lời kịch được viết theo tên làn điệu, ngoại trừ kể ra lời hát, mỗi một màn kịch các đào hát lên sân khấu thế nào rời sân khấu ra sao cùng với thần thái ngữ điệu cần phải biểu hiện đều có miêu tả.",
    "Phúc công công âm thầm lườm một cái, nhìn Bạc Nhược U nằm ở trên giường nhỏ, cảm giác có chút phức tạp khó mà diễn tả.",
    "Nhưng đêm phu nhân sinh cô nương, sáng hôm sau trời quang mây tạnh, cây hải đường trong sân cũng nở đầy hoa chỉ sau một đêm đẹp không sao tả xiết.",
    "thậm chí hoàng đế cũng bị thế gia hợp lại thay đổi không ít. Chuyện này đã xảy ra hàng trăm năm, ai nguyện để chế độ khoa cử có thể lay động nền tảng của thế gia có thể thuận lợi thi hành?",
    "Lời hát tình cảnh này cực kỳ thống khổ đau lòng, mà khiến vẻ mặt Bạc Nhược U nghiêm túc là có liên quan với miêu tả Trần Lang rời sân khấu.",
    "Vừa mở cửa, Tuyết Thanh Ninh liền thấy đập vào mắt toàn một màu trắng xóa, trên không vẫn còn tuyết rơi lả tả như bông.",

    
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