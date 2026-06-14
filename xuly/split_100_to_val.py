with open("/content/vits2-melo/melo/data/vi_speaker_44100_789/train.list", "r", encoding="utf-8") as f:
    lines = f.readlines()

val_lines = lines[:100]
train_lines = lines[100:]

with open("/content/vits2-melo/melo/data/vi_speaker_44100_789/val.list", "w", encoding="utf-8") as f:
    f.writelines(val_lines)

with open("/content/vits2-melo/melo/data/vi_speaker_44100_789/train.list", "w", encoding="utf-8") as f:
    f.writelines(train_lines)

print(
    f"Đã chuyển {len(val_lines)} dòng sang val.list, "
    f"còn lại {len(train_lines)} dòng trong train.list"
)