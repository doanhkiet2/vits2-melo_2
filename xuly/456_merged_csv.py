import shutil
with open("/content/va-data/kaggle/working/vits2-melo/melo/data/vi_speaker_44100_456/train.list", "w", encoding="utf-8") as out:
    # ghi toàn bộ file1
    with open("/content/va-data/kaggle/working/vits2-melo/melo/data/vi_speaker_44100_part4/train.list", "r", encoding="utf-8") as f1:
        shutil.copyfileobj(f1, out)


    with open("/content/va-data/kaggle/working/vits2-melo/melo/data/vi_speaker_44100_part5/train.list", "r", encoding="utf-8") as f2:
        shutil.copyfileobj(f2, out)

    with open("/content/va-data/kaggle/working/vits2-melo/melo/data/vi_speaker_44100_part6/train.list", "r", encoding="utf-8") as f3:
        shutil.copyfileobj(f3, out)