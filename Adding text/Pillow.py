from PIL import Image, ImageFont, ImageDraw
import cv2
img = Image.open('../TEST_JW/output_inpainting/airport-sign-set-vector-illustration-260nw-403667989.jpg')

# 폰트( 구글 폰트에서 이용가능 ) , 폰트크기
title_font = ImageFont.truetype('ttf/Merriweather-BoldItalic.ttf', 30)

title_text = "Hello World"

image_editable = ImageDraw.Draw(img)

# (x, y ) , ( 237, 230, 211) 색감
for x in range(3):
    image_editable.text((0,x * 50), title_text, (237, 100, 110), font=title_font)

img.save('hi.jpg')