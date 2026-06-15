# infer_batch.py

import os
import sys
import shutil
import subprocess
from pathlib import Path


MODEL = "/content/vits2-melo/logs/vi_speaker_789/G_104000.pth"
OUT_DIR = Path("/content/infer_test")
MELO_DIR = "/content/vits2-melo"
INFER_PY = "/content/vits2-melo/melo/infer.py"

TEXTS = [
      # 1
    "Hắn chậm rãi mở mắt, nhìn về phía chân trời xa xăm.",

    # 2
    "Nàng khẽ thở dài, trong lòng dâng lên một cảm giác bất an khó tả.",

    # 3
    "Trên con đường nhỏ phủ đầy sương sớm, tiếng bước chân vang lên rất nhẹ.",

    """
    Cuối hạ, cái nóng vẫn chưa tan, ánh nắng trên núi Độc Tô gay gắt chiếu xuống đỉnh đầu mọi người.

Đường núi Độc Tô hiểm trở, đá nhọn dựng đứng, hai bên vách núi như những bức tường chặn lại, ở giữa có một dòng suối từ trên cao đổ xuống, nước suối bắn tung tóe va vào các tảng đá xanh, hóa thành những hạt nước nhỏ li ti bay khắp nơi.

Một đoàn xe ngựa đang dừng lại dưới bóng râm bên dòng suối, người ngựa qua lại đang múc nước hoặc nghỉ ngơi dưới làn suối mát.

Đây là xe ngựa của gia tộc Vương ở thành Nhạc, trên đường tới đại tông môn tu tiên của Đô Châu, Thái Viêm Phái, để tham gia kỳ tuyển chọn đệ tử.

Thái Viêm Phái là một trong những tông môn lớn của Đô Châu.

Mỗi mười năm, các kỳ tài từ các thành trong châu đều tề tựu về đây, phô diễn bản lĩnh chỉ để nổi danh trong cuộc thi, từ đó bước chân vào con đường tu tiên.

Thiếu thành chủ thành Nhạc – Vương Thiệu, lúc này đang ngồi trong xe ngựa nghỉ ngơi.

Hắn năm nay mười bảy tuổi, tu vi đã đạt trung kỳ Trúc Cơ, chỉ còn một bước nữa là đến tầng thứ ba của Trúc Cơ.

Còn cách núi Cô Phùng, nơi diễn ra kỳ tuyển chọn, khoảng vài chục ngày nữa.

Trong khoảng thời gian này, nếu mỗi ngày đều dùng linh dược và linh đan để dưỡng khí, cộng thêm không ngừng khổ luyện, có lẽ hắn có thể tiến vào hậu kỳ Trúc Cơ trước khi cuộc thi bắt đầu.

Vương Thiệu chính là hy vọng lớn nhất của thành Nhạc.

Thành Nhạc chỉ là một tiểu thành vùng biên, thậm chí trên bản đồ Đô Châu cũng khó mà tìm ra.

Đã nhiều năm nay, nơi này không có một tu sĩ nào đột phá được đến Kim Đan Kỳ.

Vương Thiệu mười tuổi luyện khí, mười hai tuổi Trúc Cơ, từ tầng thứ nhất đến tầng thứ hai của Trúc Cơ đã mất năm năm.

Nếu lần này hắn có thể đột phá thành công, sẽ trở thành thiên tài đầu tiên của thành Nhạc đạt tới hậu kỳ Trúc Cơ trước mười tám tuổi.

Nếu hắn còn có thể được chọn vào Thái Viêm Phái, thành Nhạc sẽ được mở mày mở mặt với khắp thiên hạ.

Vì vậy, tất cả linh thạch, linh dược, linh đan trong thành đều được dâng lên để bồi dưỡng thiên tài này.

Bên cạnh Vương Thiệu, một thiếu nữ áo xanh xinh đẹp đang ngồi, chính là vị hôn thê của hắn, đại tiểu thư Dương Trâm Tinh của nhà họ Dương ở thành Nhạc.

Lần này, nàng đi cùng Vương Thiệu tới Thái Viêm Phái để tham dự kỳ tuyển chọn.

Dương đại tiểu thư vốn không hứng thú với tu tiên.

Tuy cùng tuổi với Vương Thiệu, nhưng nàng chỉ miễn cưỡng đạt đến sơ kỳ Luyện Khí.

Đối với nữ nhân ở thành Nhạc, so với khổ luyện tu hành, gả cho một tu sĩ làm chồng lại là lựa chọn khôn ngoan hơn.

Không chỉ được hưởng danh thơm lây, mà còn sống trong vinh hoa phú quý, được người người kính trọng.

“Thiệu ca, uống chút trà đi.”

Dương đại tiểu thư mỉm cười, nâng chén trà đưa tới bên môi Vương Thiệu.

Ánh mắt Vương Thiệu lại rơi về phía khác, hắn đứng dậy nói: “Ta ra ngoài một lát.”

Hắn vung tay áo bước xuống xe ngựa, Dương đại tiểu thư nhìn theo hướng hắn rời đi, khuôn mặt diễm lệ lập tức lộ ra vài phần dữ tợn, nghiến răng nói: “Hồ ly tinh kia!”

Thị nữ Hồng Tô tiến lại gần, vẻ mặt lo lắng hỏi: “Tiểu thư, thiếu thành chủ sẽ không định nạp cô ta làm thiếp chứ?”

“Cô ta đừng có mơ!”

Vương Thiệu đi một đoạn, dừng bước lại, nhìn thiếu nữ áo vàng đang ngồi tựa lưng vào gốc cây.

Nàng khoảng mười sáu, mười bảy tuổi, dung mạo thanh tú, làn da hơi tái nhợt, càng làm tăng thêm vẻ yếu ớt động lòng người.

Vương Thiệu nhìn nàng, bất chợt nở một nụ cười, hỏi: “Liễu cô nương, trời nóng như vậy, có muốn lên xe ngựa của ta ngồi cho mát không?”

Liễu Vân Tâm hơi sợ hãi hắn, rụt rè đáp: “Đa tạ thiếu thành chủ quan tâm, nhưng không cần đâu, ở đây rất tốt rồi.”

Liễu Vân Tâm lần này cũng đi cùng huynh trưởng Mục Tằng Tiêu để tham gia kỳ tuyển chọn của Thái Viêm Phái.

Hai người không phải huynh muội ruột, năm xưa cha mẹ của Liễu Vân Tâm từng nhận nuôi cô nhi Mục Tằng Tiêu.

Sau khi vợ chồng nhà họ Liễu qua đời, hai người nương tựa vào nhau mà sống.

Vương Thiệu đã sớm động lòng trước vẻ đẹp của Liễu Vân Tâm, chỉ là gia cảnh nàng nghèo khó, không xứng với thân phận của hắn.

Cưới nàng làm chính thê thì không thể, nhưng làm thiếp thì lại quá hợp.

Đáng tiếc, Liễu Vân Tâm không biết điều, hết lần này đến lần khác làm ngơ trước ý tốt của hắn.

Không chỉ vậy, huynh trưởng Mục Tằng Tiêu của nàng còn luôn phòng hắn như phòng trộm, khiến hắn không thể tìm được cơ hội ra tay.

Chẳng phải thế sao, hắn vừa nói vài câu với Liễu Vân Tâm, Mục Tằng Tiêu bên kia đã vội vàng chạy tới, chắn trước mặt Vương Thiệu, tức giận nói: “Vương Thiệu, ngươi định làm gì?”

Vương Thiệu nhìn thiếu niên trước mặt.

Người này mày kiếm mắt sao, ngũ quan tuấn tú.

Tuy mặc y phục vá chằng vá đụp, nhưng trong ánh mắt lại toát lên vẻ kiên định, quật cường.

Nghe nói hắn cũng từng là một thiên tài, tám tuổi luyện khí, từng được người thành Nhạc kỳ vọng, nghĩ rằng hắn sẽ trở thành một mầm non triển vọng.

Nhưng đến nay, hắn vẫn chưa thể đột phá Trúc Cơ.

“Liễu cô nương, nghe nói thân thể cô không tốt,”

Vương Thiệu không hề tức giận, trái lại còn mỉm cười nho nhã nói với Liễu Vân Tâm: “Huynh trưởng của cô tham gia kỳ tuyển chọn, chắc là vì muốn vào tông môn để lấy linh dược, linh đan chữa bệnh cho cô.

Tình cảm huynh muội sâu nặng như vậy, ta rất cảm động.
    """,
        # 4
    "Lão già chống gậy đứng lặng hồi lâu rồi mới chậm chạp quay người rời đi.",

    # 5
    "Từng tia chớp xé toạc bầu trời đen kịt, khiến đám trẻ hoảng hốt nép sát vào nhau.",

    # 6
    "Mặc dù đã chuẩn bị tâm lý từ trước, hắn vẫn không khỏi kinh ngạc khi nhìn thấy cảnh tượng trước mắt.",

    # 7
    "Ba trăm sáu mươi lăm ngày qua đi, mọi thứ dường như đã thay đổi nhưng ký ức ấy vẫn còn nguyên vẹn.",

    # 8
    "Chiếc trực thăng bay vòng qua đỉnh núi rồi nhanh chóng biến mất giữa tầng mây dày đặc.",

    # 9
    "Nếu không tận mắt chứng kiến, e rằng chẳng ai tin nổi chuyện kỳ lạ như vậy lại thực sự xảy ra.",

    # 10
    "Tiểu Thanh khẽ mím môi, ánh mắt vừa có chút do dự vừa mang theo vẻ kiên định hiếm thấy."

    
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