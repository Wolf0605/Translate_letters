import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = r'C:\Users\Wolf\PycharmProjects\Translate_letters\easy-ocr-project\patient.jpg'

img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE)

img = img[ 323:385, 163:333]

# masking code
_, mask = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
mask = cv2.bitwise_not(mask)

plt.imshow(mask)
plt.show()

# thickness
