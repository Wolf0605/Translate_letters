import cv2
import numpy as np
import matplotlib.pyplot as plt
file_path = r'dead_endjpg.jpg'
img = cv2.imread(file_path)

# cv2.imshow('hi', img)
# cv2.waitKey(0)
# plt.imshow(img)
# plt.show()
def rgb(img):
    r1, g1, b1 = img[0][0]
    r2, g2, b2 = img[-1][0]
    r3, g3, b3 = img[-1][-1]
    r4, g4, b4 = img[0][-1]

    if (r1>=0 and g1>100 and b1>100) or (r2>100 and g2>=0 and b2>100)\
            or (r3>100 and g3>100 and b3>=0) or (r4>100 and g4>100 and b4>100):
        return 0

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

return_rgb = rgb(img)
if return_rgb == 0:
    print('000')
else:
    print('nono')