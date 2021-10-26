from PIL import Image, ImageFont, ImageDraw

img = Image.open('Resources/StreetNewYork.png')

# 폰트( 구글 폰트에서 이용가능 ) , 폰트크기
title_font = ImageFont.truetype('ttf/Merriweather-BoldItalic.ttf', 150)

title_text = "Hello World"

image_editable = ImageDraw.Draw(img)

# (x, y ) , ( 237, 230, 211) 색감
image_editable.text((0,200), title_text, (237, 100, 110), font=title_font)

img.save("Newyork.png")
