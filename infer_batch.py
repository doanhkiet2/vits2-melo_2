# infer_batch.py

import os
import sys
import shutil
import subprocess
from pathlib import Path


MODEL = "/content/vits2-melo/logs/vi_speaker_789/G_120000.pth"
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
    "Tiểu Thanh khẽ mím môi, ánh mắt vừa có chút do dự vừa mang theo vẻ kiên định hiếm thấy.",
  """
  chương một

tưởng gia có hai mỹ nữ, trưởng nữ con vợ cả, quyến rũ như yêu, thứ nữ con vợ kế thanh lệ như tiên.
mẹ ruột của nàng mất sớm, đại ca tử trận nơi sa trường, đạo sĩ tính bát tự nói nàng là thiên sát cô tinh, từ nhỏ đã bị đưa vào thôn trang chịu sự đối xử lạnh nhạt của mọi người.
sau khi hồi phủ.
nàng vẫn luôn cảm động kế mẫu thương nàng như con đẻ.
muội muội thiện lương như tiên tử không để ý lời nguyền rủa mà thật lòng đối xử tốt với nàng.
trong lòng cảm kích thay muội muội vào cung.
cho rằng.
phụ thân thật lòng yêu mến.
dốc lòng phụ giúp người thương.
nàng vì vinh quang của gia tộc, hy sinh vì lợi ích toàn cục.
ai biết.
một khi thay đổi bất ngờ, người thương có được thiên hạ, mình lại phải gánh cái danh họa quốc yêu nữ.
muội muội cười thật khéo. "tỷ tỷ cũng biết mình là hạt cát trong mắt tiểu muội, hôm nay đã đến thời điểm loại bỏ hạt cát. tỷ tỷ cho rằng, đại ca và mẫu thân của tỷ vì sao mà chết? "
phụ thân mắt lạnh đối đãi, từng bước thăng chức, lúc đó nàng mới biết bản thân đã bị gia tộc từ bỏ.
bị chặt mất tứ chi biến thành người lợn, tận mắt nhìn thấy con nhỏ bị quan thần chơi đùa bỡn cợt tới chết, nàng hai mắt nhỏ máu, trong lòng lập lời thề độc.
nhưng trời xanh có mắt, cho nàng trở về mười năm trước, trưởng nữ tưởng gia, ác ma trở về!
thôi thôi thôi, nếu đã nói nàng là họa quốc yêu nữ, không bằng giáng họa cho nó long trời lở đất!
muội muội thương yêu giả nhân giả nghĩa, kế mẫu ác độc, người thương tàn nhẫn, gia tộc vô tình, diêm vương không thu mạng, nàng đến thu!
kiếp trước làm quân cờ, kiếp này thiên hạ là bàn cờ! nàng nói cười duyên dáng, xinh đẹp quyến rũ, một thân đỏ rực kiều diễm khuấy cho long trời lở đất. nàng là ác quỷ trở về từ địa ngục, nên dĩ nhiên phải khiến những kẻ đã hại nàng nợ máu trả bằng máu!
"ta muốn những kẻ đã từng nợ mạng sống của ta phải cầm mạng sống đến gặp ta, ta muốn những kẻ đã từng khinh thường ta vĩnh viễn chỉ có thể ngưỡng mộ ta, muốn trọng thần vương hầu nhìn thấy ta thì phải run rẩy, muốn đem giang sơn gấm vóc này, tất cả đều dẫm nát dưới chân! "
"ngươi là họa quốc yêu nữ, ta là loạn thần tặc tử. " hắn một thân y phục đen như mực, mắt sáng như sao. "trời sinh một đôi vừa đúng. "
cảnh đêm như mực, gió lạnh chợt nổi lên, cửa viện đổ nát bị gió thổi qua càng thêm tơi tả không chịu nổi.
mấy thô sử ma ma đang vội vàng đi trong sân, ma ma đi đầu có hơi béo, mặc bên ngoài một chiếc áo vải màu xanh, tay áo vén lên một nửa, xách một cái giỏ thức ăn, đi về phía gian phòng cuối cùng.
trong sân tràn ngập một mùi hương khác thường, ma ma tuổi nhỏ hơn một chút đi theo phía sau nhỏ giọng nói. "thật là thối, không biết lão gia cho gọi cái thứ kia làm gì, quá đáng sợ. " nói đến đây, bà ta không kiềm được mà kêu lên một tiếng, bước tới gần ma ma đi đầu, nói nhỏ vào tai. "không phải là muốn. "
"vương quý gia, nói ít vài lời đi. " ma ma áo xanh có chút phiền não. "nếu người bên ngoài nghe được sẽ không tha cho ngươi. "
vương quý gia vội vàng ngậm miệng.
dừng lại trước cửa phòng, một nha hoàn mặt tròn, tuổi còn trẻ từ bên trong đi ra, nhận lấy giỏ thức ăn của ma ma áo xanh rồi lại đi vào trong.
qua hồi lâu, nàng mang giỏ trống không trở ra. ma ma áo xanh nhận lấy, nhìn nha hoàn đó nói. "lão gia phân phó, đưa người trong phòng đi. "
"có phải là muốn. " nha hoàn mặt tròn sợ hãi.
"chúng ta không cần biết quá nhiều. " ma ma áo xanh thở dài, nói với vương quý gia. "đến đưa người đi. "
bên trong phòng thắp đèn, nhìn sáng hơn chút ít, vương quý gia bịt mũi lại, qua rất lâu mới nhìn thấy một thứ ở bên trong chậu gỗ.
vừa nhìn rõ cái vật kia, bà ta gần như muốn nôn ra. mấy ngày nay, tuy hôm nào bà ta cũng cùng với ma ma áo xanh đến đưa cơm, nhưng không thấy rõ bộ dạng người bên trong.
trong chậu gỗ, đây không thể gọi là người được nữa rồi. tứ chi của nàng đều bị chặt mất, chỉ còn một thân nguyên lành cố gắng chống đỡ trong chậu gỗ. tóc tai rối bù, bên trên còn dính một ít uế vật. lờ mờ có thể nhìn ra đây là một nữ tử.
ma ma áo xanh nhìn, trong mắt hiện lên một chút thương hại. mặc dù bà không biết rõ cô gái này là ai, nhưng dù thế thì trong tình huống này cũng khiến người ta thương xót. huống chi hôm nay lão gia đột nhiên hạ lệnh đưa người này ra ngoài, kết cục hơn phân nửa là lành ít dữ nhiều.
trong lòng vương quý gia đã hoảng sợ và buồn nôn đến cực điểm, lại không dám kháng lệnh, chỉ có thể kiên trì bưng chậu gỗ đi ra phía ngoài.
nữ tử này cũng ngoan ngoãn, không giãy dụa gào khóc. giống như đang ngủ.
theo sự phân phó bà ta đem chậu gỗ tới phòng ngủ của lão gia, trong lòng vương quý gia tự hỏi, lão gia đem thứ khiến người ta sợ hãi này đến phòng là có ý gì? bất thình lình, nữ tử trong chậu gỗ mở hai mắt ra, vương quý gia vừa vặn đụng phải.
nhắc tới cũng lạ, nữ tử kh. ủng bố đến cực điểm này lại có một đôi mắt thập phần xinh đẹp, vũ mị sinh tình, không nhiễm một hạt bụi, sáng long lanh như suối nước chảy từ giữa khe núi, lạnh như băng nhưng động lòng người.
vương quý gia sợ run rất lâu, mãi sau mới quay đầu chạy trốn khỏi phòng.
tưởng nguyễn chậm rãi mở mắt.
thời gian dài sống trong bóng tối, bây giờ đối diện với ánh sáng nàng có chút không quen. khựng lại suy nghĩ rõ tình cảnh của mình, chỉ có thể nở nụ cười mỉa mai.
nàng là đích trưởng nữ của binh bộ thượng thư, từng là nguyễn mỹ nhân, hôm nay lại bị người ta biến thành người lợn, vĩnh viễn không thể ngẩng đầu!
nàng lại nghĩ tới năm mình mười sáu tuổi, trước khi tiến cung, phụ thân nói. "nguyễn nhi, con vào cung làm phi, toàn bộ triệu gia và chúng ta đều chống lưng cho, không cần lo lắng. "
muội muội cầm chặt tay nàng, lau nước mắt. "nguyễn tỷ tỷ, tỷ là ân nhân của tố tố, cho dù có chết ta cũng khó có thể trả lại phần ân tình này. "
mà hắn, cầm tay của nàng. "hãy đợi một chút, đợi mấy ngày nữa, ta hứa sẽ thú nàng cho nàng một danh phận đàng hoàng. "
nhưng nay, phụ thân của nàng đã thăng chức tể tướng phụ quốc, đứng hàng quan nhất phẩm, kế mẫu đã là tể tướng phu nhân từ lâu, muội muội thành mẫu nghi thiên hạ, người kia đăng cơ làm hoàng đế! bọn họ vứt nàng ra khỏi đầu, thậm chí còn muốn giết nàng!
lúc năm tuổi, mẫu thân chết sớm, ca ca tử trận nơi sa trường, di nương biến thành kế thất, có vân du đạo sĩ qua đường tính ra bát tự nàng khắc phụ khắc mẫu, tưởng nguyễn bị đưa về thôn trang. mãi đến năm mười bốn tuổi cập kê, mới được niệm tình cốt nhục. tưởng quyền đưa nàng từ thôn trang về phủ. không lâu sau trong cung truyền ra tin tức, danh sách tuyển phi đợt mới có tiểu thư tưởng gia.
hoàng thượng hoài nghi tưởng gia cấu kết với bát hoàng tử, lúc này tuyển vào cung là có dụng ý khác, chẳng qua là muốn kiềm chế.
tưởng phủ chỉ có hai đích nữ, tưởng tố tố thân thể không tốt, tính cách yếu đuối đơn thuần. hoàng mệnh không thể trái, tưởng quyền ra lệnh, tưởng nguyễn tiến cung, trở thành nguyễn mỹ nhân.
mặc dù nàng nhẫn nhục chịu đựng, cũng không thể chịu đựng được phải ở dưới thân hoàng đế, đó cũng là tai hoạ khi ở trốn thâm cung, hoa bắt đầu héo rũ. nếu không vì bát hoàng tử luôn luôn cẩn thận an ủi nàng, nàng đã dùng bảy tấc lụa trắng tự vẫn trong cung từ lâu. từ nhỏ đến lớn, ngoại trừ ca ca và mẫu thân đã chết, không có ai chăm sóc chở che cho nàng, nàng giao tâm hồn thiếu nữ cho hắn. nàng bình tĩnh trở lại, cam tâm ở trong cung làm quân cờ cho hắn và tưởng gia, truyền tin tức. ai có thể ngờ tới, sau khi bức vua thoái vị, hoàng đế chết thảm, bọn họ lại nhốt nàng lại, vu oan nàng giết hoàng đế, ban cho nàng cái danh hoạ quốc yêu nữ!
lúc nàng đứng trên bậc thang, chứng kiến ánh mắt lạnh lùng của phụ thân, nàng rốt cuộc hiểu ra, nàng đã trở thành đồ bỏ! giết được thú săn, nấu đi chó săn!
bị nhốt trong ngục lao đen tối, lại được người cứu, nàng cho rằng đã tìm được đường sống, hoá ra ác mộng lại chỉ mới bắt đầu.
muội muội thanh lệ như tiên, vừa cười dịu dàng, vừa trơ mắt nhìn nàng bị người khác chặt đứt tứ chi, trờ thành người lợn.
nàng tuyệt vọng, không cam lòng, phẫn nộ, lại nghe được muội muội thanh lệ như tiên nói. "tỷ tỷ biết không, tiểu muội ngày thường vui vẻ thuần khiết, không chịu được dù chỉ là một hạt cát nhỏ bay vào mắt. hạt cát tỷ tỷ, tiểu muội đã dễ dàng bỏ qua hơn mười năm rồi, hôm nay, đã đến lúc ném hạt cát đi. "
nàng ta mỉm cười, bổ sung một câu. "bát hoàng tử, muốn lập ta làm hậu. tỷ tỷ không thể hưởng vinh quang này thì tiểu muội thay ngươi hưởng vậy. "
đau đớn thấu xương, giời nàng mới biết cái gì gọi là chết lặng. tưởng nguyễn thật sự không nghĩ ra lý do tưởng tố tố hận nàng như thế.
hình như tưởng tố tố đoán được tâm tư của nàng, cười nói. " mẫu thân của tỷ tỷ không phải là thiên kim tiểu thư phủ tướng quân sao? không phải tỷ tỷ ỷ vào cái thân phận này, không thèm để muội muội vào mắt sao? đáng tiếc a, đáng tiếc a. " nàng ta nâng má, nghiêng đầu nói. "hôm qua cả phủ tướng quân bởi vì tội danh mưu phản, trưa hôm qua đã bị bắt giam. " nàng ta nhìn chằm chằm vào tưởng nguyễn, gằn từng chữ. "cả nhà bị tịch biên tài sản, một trăm lẻ ba mạng người bị chém đầu thị chúng vào trưa hôm qua. "
tưởng nguyễn chỉ cảm thấy ngũ lôi oanh đình, đầu óc rối loạn. phủ tướng quân là nhà ngoại công của nàng, tuy năm đó mẫu thân cố ý gả cho tưởng quyền, chọc giận triệu đại tướng quân, từ đó về sau cắt đứt quan hệ, thế nhưng máu mủ tình thâm, làm sao mà lòng không đau như cắt!
nàng liều chết trừng mắt nhìn tưởng tố tố, nhưng đối phương chỉ cười mỉa mai. "tỷ tỷ hận à? đừng vội, ta còn một đại lễ muốn tặng cho tỷ, sau này tỷ gặp họ là được mà. "
vì vậy tưởng nguyễn bị đưa vào bên trong một cái phòng tối lờ mờ, đấu tranh vượt qua mấy ngày, thẳng cho đến hôm nay, mới nhìn thấy ánh sáng.
"két. " cửa vang lên một tiếng.
một nam tử béo ục ịch, cả người đầy mùi rượu, ôm một người khác ném lên giường, rồi bò sát lại người kia.
có thể loáng thoáng nhìn ra người bị ném là một đứa bé trai, đang vùng vẫy muốn thoát, đợi khi tưởng nguyễn nhìn rõ được đứa bé kia thì lập tức sợ hãi.
đó là phái nhi!
nữ tử trong cung phúc mỏng, rất nhiều người không thể sinh hạ long tử, rất nhiều long tử sinh ra nhưng lại chết đi. mâu thân thân sinh của phái nhi chỉ là một tiểu cung nữ, sinh hạ phái nhi rồi chết đi. hoàng thượng cũng không nhìn người con trai xuất thân thấp hèn này, ngày đó cũng không biết xảy ra chuyện gì, chỉ biết đứa bé này được giao cho nàng nuôi dưỡng.
sáu năm trôi qua, nàng cùng phái nhi sớm đã có tình cảm mẫu tử. từ lúc cung biến, nàng liền sai cung nữ thiếp thân của mình ôm phái nhi chạy trốn, nhưng hoá ra vẫn trốn không thoát.
"mẫu thân! mẫu thân! " phái nhi giãy dụa gào khóc, trốn không khỏi đôi tay đang sờ loạn trên người mình.
tưởng nguyễn chỉ cảm thấy toàn thân lạnh buốt, trường tương hầu lý đống thích đùa giỡn nam hài, từ lúc vào cung nàng đã sớm biết. nhưng mà hôm nay, nàng lại chỉ có thể trơ mắt nhìn con của mình bị ác ma này làm nhục.
nàng kêu lên thật to, nhưng chỉ có thể phát ra tiếng "a. a" khàn đặc.
lý đống chán ghét nhìn xuống. "kêu thì được cái gì, nương nương không thể tới đây cứu ngươi đâu, ngoan ngoãn chút đi. "
hắn nghĩ nghĩ, lại vẫn hèn hạ khuất phục hoàng uy, không dám có động tác gì khác, chuyên tâm đùa giỡn nam hài trước mặt.
tưởng nguyễn ngồi ở trong chậu gỗ, đến lúc này, nàng mới biết vì sao tưởng tố tố chừa lại một đôi mắt cho nàng, nàng ta muốn nàng tận mắt nhìn thấy người thân chết ở trước mặt mình.
nàng như một tượng gỗ sững sờ ngồi ở trong chậu, từng mảng ký ức xẹt qua trước mắt, vẻ mặt của mẫu thân trước khi chết, phụ thân lạnh nhạt, bát hoàng tử hứa hẹn, tưởng tố tố nắm tay của nàng nói lời cảm tạ, hoàng thượng đối xử thờ ơ, hậu cung khổ sở, cuối cùng biến thành phái nhi đang giãy dụa kêu khóc.
lý đống trong lúc lơ đãng quay đầu lại, thình lình trông thấy người trong chậu gỗ, hoảng sợ ngã xuống giường, kêu to. "có ai không, có ai không? ?? "
nữ tử trong chậu gỗ, thần sắc đờ đẫn, hai hàng huyết lệ chảy qua đôi má, cứ thế như gột rửa mà toát ra cảm giác thê lương. gia đinh phá cửa đi vào, nhất thời cũng giật mình đứng nguyên tại chỗ, chỉ cảm thấy như ác quỷ địa ngục đến lấy mạng, toàn thân lạnh buốt.
lý đồng hổn hển nói. "còn đứng ngây đó làm gì, loạn côn đánh chết cho ta. " dưới sự sợ hãi, hắn đã sớm vứt mệnh lệnh của vị nương nương kia ra khỏi đầu, dù sao trong viện đều là người của hắn, không lo sẽ để lại tai tiếng.
gia đinh phục hồi lại tinh thần, nắm chặt gậy trong tay tiến lên, không quan tâm mà đánh xuống.
không có người nghe được, trong lòng người trong chậu gỗ nguyền rủa. "cho dù vĩnh viễn không siêu sinh, tan thành mây khói, cũng chỉ nguyện biến thành lệ quỷ! để cho những người hại ta nợ máu trả bằng máu! "
cùng lúc đó, trong điện dương bình.
"hôm nay hoàng thượng hình như không có tinh thần. " tưởng tố tố khẽ cười nói.
tân đế nâng mắt nhìn nữ tử đối diện, mũ phượng khăn quàng vai, mặt mày tinh xảo, một thân trang phục cao quý càng tôn lên dáng vẻ không giống phàm nhân, như là tiên nữ trên chín tầng trời. nữ nhi tưởng quyền này quả thật là thanh lệ thoát tục.
"vẫn chưa có tin tức của tưởng nguyễn hay sao? " hắn bất ngờ thấp giọng hỏi.
sắc mặt tưởng tố tố buồn bã. "không có, tỷ tỷ chắc là mang theo phái nhi chạy thoát rồi, những năm gần đây tỷ ấy cũng cực khổ, nhưng cho dù như thế nào cũng nên tin hoàng thượng mới đúng. "
tân đế nghĩ đến tưởng nguyễn, lại phát hiện dù có cố nhớ lại như thế nào, tưởng nguyễn trong ấn tượng của hắn cũng chỉ là một cái bóng mơ hồ. thanh danh của nàng không tốt, nhiều nhất cũng chỉ là một nữ nhân có sắc đẹp mà thôi, hắn cần chính là thế lực sau lưng tưởng gia, tưởng nguyễn hay tưởng tố tố cũng không có gì khác nhau. tưởng nguyễn đã là nữ nhân của tiên hoàng, hắn tuyệt đối sẽ không lấy.
tuy tưởng nguyễn đã bỏ đi, nhưng hắn vẫn còn có chút chần chừ, ở trong cung nhiều năm như vậy, rất nhiều lúc đều là dựa vào tưởng nguyễn để vượt qua hiểm cảnh, nàng đã giúp mình không ít. thế nhưng vì sao không đợi hắn hạ quyết định, lại trốn đi trước.
hắn không thích cảm giác không khống chế được này,
hừ lạnh một tiếng, tân đế nói. "không biết tốt xấu. thời cơ đã tới, đi thôi. "
tưởng tố tố dạ một tiếng, để tay vào trong lòng bàn tay nam tử.
tuyên đức năm thứ mười tám, tân hoàng đăng cơ, lập tưởng thị làm hậu, tự mình lên ngôi, nguyện vĩnh viễn một lòng.
  """
    
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