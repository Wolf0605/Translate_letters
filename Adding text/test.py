from PIL import Image ,ImageFont, ImageDraw, ImageOps


into = 'Resources/final-cover-forest.jpg'
# What's at?
def draw_text_90_into (text: str, into, at: list):
#     # Measure the text area
    font = ImageFont.truetype ('ttf/Merriweather-BoldItalic.ttf', 100)
    # wi = x, hi = y
    wi, hi = font.getsize(text)
    print(wi, hi)

    # Copy the relevant area from the source image
    # crop(left, top, right, bottom)
    img = into.crop ((at[0], at[1], at[0] + hi, at[1] + wi))

    # Rotate it backwards ( 시계 반대 방향 )
    img = img.rotate (270, expand = 1)

    # Print into the rotated area
    d = ImageDraw.Draw(img)

    # fill = color*
    d.text ((0, 0), text, font = font, fill = (0, 0, 0))
    # # # #
    # # # # # Rotate it forward again (시계 반대 방향 )
    img = img.rotate (90, expand = 1)

    # Insert it back into the source image
    # Note that we don't need a mask
    into.paste (img, at)
    # img.show()

def draw_text(text: str, into, at: tuple):

    f = ImageFont.truetype ('ttf/Merriweather-BoldItalic.ttf', 100)
    # image.new( mode, size(A x B))
    txt=Image.new('L', (500,500))
    d = ImageDraw.Draw(txt)
    d.text( at, text,  font=f, fill=100)

    w=txt.rotate(0,  expand=1)

    into.paste(ImageOps.colorize(w, (0,0,0), (255, 255, 84)), (242,60), w)
    into.show()

draw_text('hi wolf', into, (0,0))




# Rotate Image By 180 Degree
# rotated_image1 = into.rotate(10)
#
# # This is Alternative Syntax To Rotate
# # The Image
# rotated_image2 = into.transpose(Image.ROTATE_90)
#
# # This Will Rotate Image By 60 Degree
# rotated_image3 = into.rotate(60)
#
# rotated_image1.show()
# # rotated_image2.show()
# # rotated_image3.show()
# rotated_image1.save('rolex.png')