import os
import re

# Thư mục cần xử lý
ROOT_DIR = "/content/va-data/kaggle/working/vits2-melo/melo/data/vi_speaker_44100_456/wavs"

# abc copy(3)__xyz -> abc3__xyz
pattern = re.compile(r"\s*-\s*Copy\s*\((\d+)\)")

for root, dirs, files in os.walk(ROOT_DIR):
    for filename in files:
        new_name = pattern.sub(r"\1", filename)

        if new_name != filename:
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_name)

            # Tránh ghi đè nếu file đích đã tồn tại
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
            else:
                print(f"Skipped (exists): {new_name}")