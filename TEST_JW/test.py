import cv2
import matplotlib.pyplot as plt
import easyocr
import sys
import googletrans
from typing import List
import requests
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
from sklearn.cluster import KMeans
import textwrap

# print(torch.cuda.is_available())

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# 이미지 파일 경로
file_path = r'airport-sign-set-vector-illustration-260nw-403667989.jpg'
img = cv2.imread(file_path, cv2.IMREAD_COLOR)

# secret key 로 집어넣어야함
CLIENT_ID = "MawiiHEojSbWlRvZjWEM"
CLIENT_SECRET = "gY1PNWHP54"

if img is None:
    print('Image load failed!')
    sys.exit()


# 이미지 출력함수
def display(img):
    # img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    plt.figure(figsize=(15, 15))
    plt.imshow(img)
    plt.show()


def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


def easy_ocr_result(img, language='en', draw=True, text=False):
    reader = easyocr.Reader([language])
    results = reader.readtext(img)

    # 바운딩박스 리스트
    bbox_list = []
    # 텍스트 리스트
    text_list = []

    if draw == False: # 원래 이미지만 출력
        display(img)

    elif draw == True and text == False: # 이미지에 바운딩 박스그리기
        img2 = img.copy()
        # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        for (bbox, text, prob) in results:
            # display the OCR'd text and associated probability
            # print("[INFO] {:.4f}: {}".format(prob, text))

            bbox_list.append(bbox)
            text_list.append(text)
            # unpack the bounding box
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            # cleanup the text and draw the box surrounding the text along
            # with the OCR's text itself
            cv2.rectangle(img2, tl, br, (255, 0, 0), 2)

        # show the output image
        display(img2)

    elif draw == True and text == True:  # 이미지에 바운딩 + 인식한 글자
        img2 = img.copy()
        # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        for (bbox, text, prob) in results:
            # display the OCR'd text and associated probability
            # print("[INFO] {:.4f}: {}".format(prob, text))

            bbox_list.append(bbox)
            text_list.append(text)

            # unpack the bounding box
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            # cleanup the text and draw the box surrounding the text along
            # with the OCR'd text itself
            text = cleanup_text(text)
            cv2.rectangle(img2, tl, br, (255, 0, 0), 2)
            cv2.putText(img2, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # show the output image
        display(img2)
    return bbox_list, text_list

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def array_box(bbox_list, text_list):
    text_num = [x for x in range(len(text_list))]
    try:
        for x in text_num:
            start = (bbox_list[x][0][1] + bbox_list[x][3][1]) //2
            later = (bbox_list[x+1][0][1] + bbox_list[x+1][3][1]) //2
            t = 10
            if start - t <= later <= start + t:
                bbox_list[x+1][0][1] = bbox_list[x][0][1]
                text_num.remove(x + 1)
            else:
                pass
    except:
        pass
    # 하나씩 집어 넣어서 sort 정렬.
    sort_idx = []
    for x in range(len(text_list)):
        t = bbox_list[x][0][0] + bbox_list[x][0][1] * 10000
        sort_idx.append(t)
    # print(sort_idx)
    sorted_index = argsort(sort_idx)
    return sorted_index

def sum_box(box_list, ttext_list, sorted_index):
    text_num = [x for x in range(len(ttext_list))]
    bbox_list = []
    text_list = []
    for x in sorted_index:
        bbox_list.append(box_list[x])
        text_list.append(ttext_list[x])

    try:
        for x in text_num:
            start = bbox_list[x][1]
            later = bbox_list[x+1][0]
            t= 20
            if start[0] - t <= later[0] <= start[0] + t and start[1] -t <= later[1] <= start[1] +t:
                bbox_list[x] = [bbox_list[x][0],bbox_list[x+1][1],bbox_list[x+1][2],bbox_list[x][3]]
                text_list[x] = text_list[x] + ' ' +text_list[x+1]
                bbox_list.remove(bbox_list[x+1])
                text_list.remove(text_list[x+1])
                text_num.remove(x+1)
            else:
                pass
    except:
        pass
    return np.array(bbox_list), text_list

x = 1
z = 0
y=[]
start_index = []
change_start_index = []
len_index=[]
def jaegi(bbox_list, text_list, x, z):
    t = 10
    if x == len(bbox_list):
        if len(start_index) == 0:
            pass
        elif len(start_index) == 1:
            len_index.append(len(y))
        else:
            for x in range(len(start_index)-1):
                len_index.append(y.index(start_index[x+1])-y.index(start_index[x])+1)
            len_index.append(len(y[y.index(start_index[-1]):])+1)
        return text_list, change_start_index, len_index
    elif bbox_list[x-1][0][0] - t <= bbox_list[x][0][0] <= bbox_list[x-1][0][0] + t\
            and bbox_list[x][0][1] - t <= bbox_list[x-1][3][1] <= bbox_list[x][0][1] + t:
        text_list[z] = text_list[z] +' ' + text_list[z+1]
        # x 불러오기,
        text_list.remove(text_list[z+1])
        # print(text_list)
        y.append(x-1)
        if len(y) == 0:
            pass
        elif len(y) == 1:
            start_index.append(y[0])
            change_start_index.append(y[0])
        elif y[-1] != y[-2]+1:
            start_index.append(y[-1])
            change_start_index.append(start_index[-2] + y[-1] - (y[-2] + 1))

        x += 1
        # print('x :', x)
        return jaegi(bbox_list, text_list, x, z)
    else:
        x += 1
        z += 1
        return jaegi(bbox_list, text_list, x, z)

def translate_texts(texts: List[str], type='google') -> List[str]:
    global tranlated_texts
    if type == 'google':
        translator = googletrans.Translator()
        tranlated_texts = [
            translator.translate(text=text, src='en', dest='ko').text
            for text in texts
        ]
    elif type == 'naver':
        url = "https://openapi.naver.com/v1/papago/n2mt"
        header = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
        tranlated_texts = []
        for text in texts:
            data = {'text': text, 'source': 'en', 'target': 'ko'}
            response = requests.post(url, headers=header, data=data)
            rescode = response.status_code
            if rescode == 200:
                t_data = response.json()
                tranlated_texts.append(t_data['message']['result']['translatedText'])
            else:
                print("Error Code:", rescode)

    return tranlated_texts

def split_text(tranlated_texts, change_start_index, len_index):
    if change_start_index == []:
        pass
    else:
        s = 0
        for idx, x in enumerate(change_start_index):
            line = round (len(tranlated_texts[x+s]) / len_index[idx])
            # print(type(tranlated_texts[x]))
            # print(line, type(line) )
            # print(len_index[idx], type(len_index[idx]))
            # print('line :', line)
            split_word = textwrap.wrap(tranlated_texts[x+s], line)

            if len(split_word) != len_index[idx]:
                split_word [-2] = split_word[-2] + split_word[-1]
                del split_word[-1]
            else:
                pass
            tranlated_texts.remove(tranlated_texts[x+s])
            s -= 1
            for k in split_word:
                tranlated_texts.insert(change_start_index[idx] + s+1,k)
                s += 1
    return tranlated_texts







def cut_image(img, bbox):
    x_min = int(bbox[0, 0])
    x_max = int(bbox[1, 0])
    y_min = int(bbox[0, 1])
    y_max = int(bbox[2, 1])

    img = img[y_min:y_max, x_min:x_max]

    return img

def mask_image(img2):
    # masking 작업
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

    # 글씨 색이 밝든 어둡든 masking 씌워주기
    return_rgb = rgb(img2)
    if return_rgb != 0:
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, 3)
    # img_contour = cv2.drawContours(img_original, contours, -1, (0, 255, 0), 3)

    # plt.imshow(mask)
    # plt.show()
    return mask


def change_original(img, masked_img, bbox):
    x_min = int(bbox[0, 0])
    x_max = int(bbox[1, 0])
    y_min = int(bbox[0, 1])
    y_max = int(bbox[2, 1])

    img[y_min:y_max, x_min:x_max] =  masked_img
    return img

# 123
def rgb(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    flat_list = list(mask.ravel())
    if flat_list.count(0) > len(flat_list)/2:
        return 0

# def rewrite(tranlated_texts ,bbox_list, color, index):
#     # 폰트( 구글 폰트에서 이용가능 ) , 폰트크기
#
#     image_path = 'output_inpainting'
#     img = Image.open(f"{image_path}/{file_path}")
#     image_editable = ImageDraw.Draw(img)
#
#     # (x, y ) , ( 237, 230, 211) 색감
#     bbox_hi = []
#     for bbox in bbox_list:
#         bbox_hi.append(bbox[2][1] - bbox[0][1])
#
#     bbox_hi_median = int(np.median(bbox_hi))
#     bbox_hi_median_diff = np.abs(np.array(bbox_hi) - bbox_hi_median)  # 절대값 추출
#     # print('bbox_hi_median',bbox_hi_median)
#     hi_lt_idx = np.where(bbox_hi_median_diff < 10)  # 10보다 적게나는 값 idx 추출
#
#     # bbox_hi > array 변경
#     bbox_hi = np.array(bbox_hi)
#     # 차이 작은 값은 median값으로 변경 큰것은 그대로.
#     bbox_hi[hi_lt_idx] = bbox_hi_median
#
#     # print('bbox_hi', bbox_hi)
#     for idx, (bbox, color) in enumerate(zip(bbox_list, color_list)):
#         # print('fontsize',bbox_hi[idx]-15)
#         text = tranlated_texts[idx]
#
#         title_font = ImageFont.truetype("ttf/NotoSansKR-Bold.otf", np.maximum(2, int(bbox_hi[idx]) - 5))  # -가 될경우 최소 2로 설정.
#         wi, hi = title_font.getsize(text)
#
#         print('bbox_hi[idx]-15', type(bbox_hi[idx] - 15))
#
#         print('title_font', title_font)
#         start_x = ((bbox[0][0] + bbox[1][0]) // 2 - wi / 2)
#         start_y = ((bbox[0][1] + bbox[2][1]) // 2 - hi / 2)
#         image_editable.text((start_x, start_y), text, color, anchor=None, font=title_font)
#     save_rewrite_images(img)

def rewrite(tranlated_texts, bbox_list, color_list):
    image_path = 'output_inpainting'
    img = Image.open(f"{image_path}/{file_path}")

    image_editable = ImageDraw.Draw(img)

    ### font size ###
    bbox_hi = []
    for bbox in bbox_list:
        bbox_hi.append(bbox[2][1] - bbox[0][1])

    bbox_hi_median = int(np.median(bbox_hi))
    bbox_hi_median_diff = np.abs(np.array(bbox_hi) - bbox_hi_median)  # 절대값 추출
    # print('bbox_hi_median',bbox_hi_median)
    hi_lt_idx = np.where(bbox_hi_median_diff < 10)  # 10보다 적게나는 값 idx 추출

    # bbox_hi > array 변경
    bbox_hi = np.array(bbox_hi)
    # 차이 작은 값은 median값으로 변경 큰것은 그대로.
    bbox_hi[hi_lt_idx] = bbox_hi_median

    # print('bbox_hi', bbox_hi)
    for idx, (bbox, color) in enumerate(zip(bbox_list, color_list)):
        # print('fontsize',bbox_hi[idx]-15)
        text = tranlated_texts[idx]

        title_font = ImageFont.truetype("ttf/NotoSansKR-Bold.otf", np.maximum(2, int(bbox_hi[idx]) - 5))  # -가 될경우 최소 2로 설정.
        wi, hi = title_font.getsize(text)

        print('bbox_hi[idx]-15', type(bbox_hi[idx] - 15))

        print('title_font', title_font)
        start_x = ((bbox[0][0] + bbox[1][0]) // 2 - wi / 2)
        start_y = ((bbox[0][1] + bbox[2][1]) // 2 - hi / 2)
        image_editable.text((start_x, start_y), text, color, anchor=None, font=title_font)

    save_rewrite_images(img)

def decsion_font_size( bbox_hi, text):
    font_size = 1
    title_font = ImageFont.truetype('ttf/NotoSansKR-Bold.otf', font_size)
    _, hi = title_font.getsize(text)
    while hi < bbox_hi:
        title_font = ImageFont.truetype('ttf/NotoSansKR-Bold.otf', font_size)
        font_size += 1
        _, hi = title_font.getsize(text)
    return font_size



def save_inpainting_images():
    image_path = f'output_inpainting'
    save_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    save_img.save(f"{image_path}/{file_path}")

def save_rewrite_images(img):
    image_path = 'output_rewrite'
    img.save(f"{image_path}/{file_path}")

def clt_(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1], 3)) # height, width 통합

    k = 3 # 예제는 5개로 나누겠습니다
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

def contour_mask(mask):
    contour_pos = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, 3)

    for pos in range(len(contours)):
        area = cv2.contourArea(contours[pos])
        if area > 100:
            contour_pos.append(pos)

def change_color(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray 영상으로 만들기
    ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU) # 마스킹

    if len(img_binary[img_binary > 250]) > len(img_binary[img_binary < 250]):
        img_binary = cv2.bitwise_not(img_binary)

    masked = cv2.bitwise_and(img, img, mask = img_binary)


    b, g, r = cv2.split(masked)
    b, g, r = int(np.mean(b[b > 0])), int(np.mean(g[g > 0])), int(np.mean(r[r > 0]))

    return b,g,r


def change_bg_color(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray 영상으로 만들기
    ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)  # 마스킹
    bg_binary = cv2.bitwise_not(img_binary)

    if len(img_binary[img_binary > 250]) > len(img_binary[img_binary < 250]):
        bg_binary = cv2.bitwise_not(bg_binary)

    masked_bg = cv2.bitwise_and(img, img, mask=bg_binary)

    b, g, r = cv2.split(masked_bg)
    b, g, r = int(np.mean(b[b > 0])), int(np.mean(g[g > 0])), int(np.mean(r[r > 0]))

    a = np.ones(shape=img.shape, dtype=np.uint8)
    b = a[:, :, 0] * b
    g = a[:, :, 1] * g
    r = a[:, :, 2] * r

    return b, g, r
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bbox_list, text_list = easy_ocr_result(img)
    sorted_index = array_box(bbox_list, text_list)
    print(' len - bbox_list : ', len(bbox_list))
    bbox_list, text_list = sum_box(bbox_list, text_list, sorted_index)
    print(' len - bbox_list : ', len(bbox_list))
    text_list, change_start_index, len_index = jaegi(bbox_list, text_list, x, z)
    tranlated_texts: List[str] = translate_texts(texts=text_list, type='naver')
    # print(f'Tranlated_texts : {tranlated_texts}')
    tranlated_texts = split_text(tranlated_texts, change_start_index, len_index)
    print(tranlated_texts)
    # print(tranlated_texts)

    color_list = []
    # reverse_sorted_index = []
    for bbox in bbox_list:
        img_cut = cut_image(img, bbox)
        color_list.append(change_color(img_cut))
        b, g, r = change_bg_color(img_cut)
        # print(b.shape)
        img_cut[:, :, 0] = b
        img_cut[:, :, 1] = g
        img_cut[:, :, 2] = r
        # version _ 1
        # clt = clt_(img_cut)
        # hist = centroid_histogram(clt)
        # color_list.append(clt.cluster_centers_)
        # reverse_index = (-hist).argsort()
        # reverse_sorted_index.append(reverse_index)
        #
        # mask = mask_image(img_cut)
        # masked_img = cv2.inpaint(img_cut, mask, 3, cv2.INPAINT_TELEA)
        img = change_original(img,img_cut, bbox)

    save_inpainting_images()

    rewrite(tranlated_texts,bbox_list,color_list)

