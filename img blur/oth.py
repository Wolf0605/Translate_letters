import cv2
import matplotlib.pyplot as plt
import numpy as np
def blur(img_path):
  img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE)
  img = img[ 323:385, 163:333]
  dst2 = np.zeros(img.shape, np.uint8)
  bw = img.shape[1] // 4
  bh = img.shape[0] // 4
  for y in range(4):
    for x in range(4):
      src_ = img[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
      dst_ = dst2[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
      cv2.threshold(src_, 0, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU, dst_)
  img_rgb = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
  plt.imshow(img_rgb)
  plt.show()

blur(r'C:\Users\Wolf\PycharmProjects\Translate_letters\easy-ocr-project\patient.jpg')