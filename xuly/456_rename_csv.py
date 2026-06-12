import re

input_file = "/content/va-data/kaggle/working/vits2-melo/melo/data/vi_speaker_44100_456/train.list"
output_file = "/content/va-data/kaggle/working/vits2-melo/melo/data/vi_speaker_44100_456/train2.list"

# copy_pattern = re.compile(r"\s*-\s*Copy\s*\((\d+)\)", re.IGNORECASE)

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:
        # Tách phần path khỏi các cột còn lại
        parts = line.rstrip("\n").split("|")

        path = parts[0]

        # /kaggle/working -> /content
        path = path.replace("/kaggle/working", "/content")

        # part4/5/6 -> 456
        path = path.replace(
            "vi_speaker_44100_part4",
            "vi_speaker_44100_456"
        )
        path = path.replace(
            "vi_speaker_44100_part5",
            "vi_speaker_44100_456"
        )
        path = path.replace(
            "vi_speaker_44100_part6",
            "vi_speaker_44100_456"
        )

        # " - Copy (9)" -> "9"
        # path = copy_pattern.sub(r"\1", path)

        parts[0] = path

        fout.write("|".join(parts) + "\n")

print("Done!")