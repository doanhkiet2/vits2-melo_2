import csv
import re

pattern = re.compile(r"\s*copy\((\d+)\)")

with open("/content/va-data/kaggle/working/vits2-melo/melo/data/vi_speaker_44100_456/train_raw.list", newline="", encoding="utf-8") as fin, \
     open("/content/va-data/kaggle/working/vits2-melo/melo/data/vi_speaker_44100_456/train.list", "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)

    writer.writeheader()

    for row in reader:
        row["path"] = pattern.sub(r"\1", row["path"])
        writer.writerow(row)