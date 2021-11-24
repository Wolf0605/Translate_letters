import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
file_path = r'DDEEEDDD.png'
img = cv2.imread(file_path)
img_width = img.shape[1]
img_height = img.shape[0]
def rgb(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    flat_list = list(mask.ravel())
    if flat_list.count(0) > len(flat_list)/2:
        return 0

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

def call():
    file_path = r'dead_endjpg.jpg'
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)

    save_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    save_img.save("image_test.jpg")

    cv2.imshow('img', img)
    cv2.waitKey(0)

def clt_(img):
    img = img.reshape((img.shape[0] * img.shape[1], 3)) # height, width 통합

    k = 5 # 예제는 5개로 나누겠습니다
    clt = KMeans(n_clusters = k, random_state=0)
    clt.fit(img)
    return clt



def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_))+1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist

def draw_contour(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    return_rgb = rgb(img)
    if return_rgb != 0:
        thresh = cv2.bitwise_not(thresh)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, 3)

    index_box = []
    for x in range(len(contours)):
        if 0 not in contours[x]:
            for i in range(len(contours[x])):
                if img_width-5 <= contours[x][i][0][0] or img_height-5 <= contours[x][i][0][1]:
                   break
                else:
                    index_box.append(x)
    new_conoturs = []
    for x in index_box:
        new_conoturs.append(contours[x])

    img_contour = cv2.drawContours(img, new_conoturs, -1, (0, 255, 0), thickness=cv2.FILLED)

    return img_contour
# img_contour = draw_contour(img)
# cv2.imshow('gg', img_contour)
# cv2.waitKey(0)
lst = [[[1,2],[3,4],[5,6],[7,8]]]
print(lst[0][0])
s, y = zip(lst[0][0])
print(s,y)