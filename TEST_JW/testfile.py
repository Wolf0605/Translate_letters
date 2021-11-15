import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
file_path = r'dead_endjpg.jpg'
img = cv2.imread(file_path)

def rgb(img):
    r1, g1, b1 = img[0][0]
    r2, g2, b2 = img[-1][0]
    r3, g3, b3 = img[-1][-1]
    r4, g4, b4 = img[0][-1]

    if (r1>=0 and g1>100 and b1>100) or (r2>100 and g2>=0 and b2>100)\
            or (r3>100 and g3>100 and b3>=0) or (r4>100 and g4>100 and b4>100):
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

img = img.reshape((img.shape[0] * img.shape[1], 3)) # height, width 통합

k = 5 # 예제는 5개로 나누겠습니다
clt = KMeans(n_clusters = k)
clt.fit(img)

for center in clt.cluster_centers_:
    print(center)

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


hist = centroid_histogram(clt)

print(hist)
def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

bar = plot_colors(hist, clt.cluster_centers_)


# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()