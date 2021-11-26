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

def draw_contour(img,result):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    return_rgb = rgb(img)
    if return_rgb != 0:
        thresh = cv2.bitwise_not(thresh)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=2)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, 3)

    index_box = []
    for x in range(len(contours)):
        if 0 not in contours[x]:
            for i in range(len(contours[x])):
                if img_width-10 <= contours[x][i][0][0] or img_height-10 <= contours[x][i][0][1]:
                   break
                else:
                    index_box.append(x)
    index_box = set(index_box)
    new_conoturs = []

    for x in index_box:
        new_conoturs.append(contours[x])

    mask = cv2.drawContours(img, new_conoturs, -1, tuple(result), thickness=cv2.FILLED)

    return thresh, mask

def return_img(img):
    x_min = 0
    x_max = 100
    y_min = 0
    y_max = 100

<<<<<<< HEAD
    return img_contour
# img_contour = draw_contour(img)
# cv2.imshow('gg', img_contour)
# cv2.waitKey(0)
def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)
def tt():
    text_list = [[[0,0],[13,0]], [[0,1],[13,1]], [[14,0],[27,0]]]
    sort_idx = []
    for x in range(len(text_list)):
        t = text_list[x][0][0] + text_list[x][0][1] * 10000
        sort_idx.append(t)
    print(sort_idx)

    print(argsort(sort_idx))
lst = []
def ja(lst):
    for x in range(5):
        if x in [1,3,5]
            lst.append(x)
        else:
=======
    quarter_y = int(y_max - (y_max - y_min) / 4)
    crop_img = img[quarter_y:y_max, x_min:x_max]
    img = img[y_min:y_max, x_min:x_max]
    return crop_img, img

crop_img, img = return_img(img)
img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
cv2.imshow('gg', img)
cv2.waitKey(0)

# color_list = []
# reverse_sorted_index = []
#
# clt = clt_(img)
# hist = centroid_histogram(clt)
# color_list.append(clt.cluster_centers_)
# reverse_index = (-hist).argsort()
# reverse_sorted_index.append(reverse_index)
# result = list(map(int, color_list[0][reverse_sorted_index[0][0]]))
#
# t, img_contour= draw_contour(img, result)
# cv2.imshow('gg', img_contour)
# cv2.waitKey(0)
>>>>>>> c4e17e2698f802e435584f69a0c4a9d5843791e3
