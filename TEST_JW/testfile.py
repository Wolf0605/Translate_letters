import cv2
import numpy as np
import matplotlib.pyplot as plt
file_path = r'sign_noun_002_33753.jpg'
img = cv2.imread(file_path)
print(img.shape)
print(img[0][0])

def rgb(img):
    r,g,b = img[0][0]
    if r>100 & g>100 & b>100:
        return 'bright'
    else:
        return 'dark'
def hi():
    if 'dark':
        print('dddddark')
    else:
        print('brigh2222t')
# rgb(img)
# hi()

def mask_image(img):
    # masking 작업
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

    # 색상 검출해서 글씨 색이 밝든 어둡든 masking 씌워주기
    # 글이 어두운색 이면,
    # if 'dark':
    #     mask = cv2.bitwise_not(mask)
    mask = cv2.bitwise_not(mask)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    plt.imshow(mask)
    plt.show()
    return mask
mask_image(img)