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
file_path = r'am3.png'
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

# 랜덤으로 배열되어 있는 text_list 와 bbox_list를 사람이 글을 읽듯 그 순서에 맞게 재배열해주는 과정
def array_box(bbox_list, text_list):
    t = 10
    # 첫번쨰 바운딩박스의 1번쨰점의 y 값과 3 번쨰점의 y 값의 평균을 구하는것
    # 그리고 아래도 같은 작업을 반복하여 만약 차가 +- 10 정도 나오면 0번쨰 y 값이 같도록 설정한다
    for x in range(len(text_list)):
        start = (bbox_list[x][1][1] + bbox_list[x][3][1]) //2
        for y in range(len(text_list)):
            later = (bbox_list[y][1][1] + bbox_list[y][3][1]) //2
            if start - t <= later <= start + t:
                bbox_list[y][0][1] = bbox_list[x][0][1]
    # 하나씩 집어넣어서 y 값을 같도록하였으니까 x 같은 y 값에 x 만 다르므로 , x 에 따라 순서가 배열되도록 설정
    sort_idx = []
    for x in range(len(text_list)):
        t = bbox_list[x][0][0] + bbox_list[x][0][1] * 10000
        sort_idx.append(t)
    # argsort 를 활용하여 그 순서에 맞게 bbox_list 와 text_list 를 재배열
    sorted_index = argsort(sort_idx)
    box_list = []
    ttext_list = []
    for x in sorted_index:
        box_list.append(bbox_list[x])
        ttext_list.append(text_list[x])
    return box_list, ttext_list

# 바운딩박스 옆에 바로 바운딩박스가 붙어있는 경우, 그걸 하나의 박스로 만들고 , text_list도 두개 혹은 그 이상을 하나로 만들어 주는 과정
# 몇줄이 나오는지 모르니까 재귀를 이용한다
x = 0
def sum_box(bbox_list, text_list,x):
    # 처음 바운딩박스와 그다음 바운딩 박스의 0번쨰 점을 start, later 로 지정
    try:
        start = bbox_list[x][0]
        later = bbox_list[x+1][0]
        t = 10
        # start 와 later 의 y 값이 같고( array_box 에서 이미 y 를 같게 만들어줬다) , later 의 x 값이
        # 바운딩박스의 첫번쨰점의 x 값과 +- 10 차이가 나면 bbox_list 와 text_list를 ex) bbox_list[0] + bbox_list[1]
        # text_list[0] + text_list[1] 을 더하고 원래 있던 bbox_list[1], text_list[1] 은 삭제하는 형식
        if start[1] == later[1] and bbox_list[x][1][0] -t <= later[0] <= bbox_list[x][1][0] +t:
            bbox_list[x] = [bbox_list[x][0], bbox_list[x + 1][1], bbox_list[x + 1][2], bbox_list[x][3]]
            text_list[x] = text_list[x] + ' ' + text_list[x + 1]
            bbox_list.remove(bbox_list[x + 1])
            text_list.remove(text_list[x + 1])
            return sum_box(bbox_list, text_list, x)
        else:
            x += 1
            return sum_box(bbox_list,text_list, x)
    except:
        return np.array(bbox_list), text_list

x = 1
z = 0
y=[]
start_index = []
change_start_index = []
len_index=[]

# 한 문단을 어떻게 한문단으로 인식할지 지정하는 코드. 보통 한 문단이라고 하는것은, 사람이 글을 읽을 때와 마찬가지로 시작점의 x 값이 비슷하며,
# 글의 문장과 다음 문장의 시작점 x 와 글이 그 글씨의 높이가 일정한 경우. 그래서 하나의 문단을 전체 하나의 문장으로 뽑는다.
def jaegi(bbox_list, text_list, x, z):
    t = 10
    # 만약 맨 마지막 호출 x == len(bbox_list) 일떄, len_index 각 문단이 몇줄인지 저장
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
    # 한 문단의 시작점 x 가 비슷하고, y 가 비슷할경우, text_list 를 합쳐주고 위와같이 뒷부분을 삭제시키는 함수
    # 그리고 문단이 시작되는 index 를 change_start_index 에 집어넣고 이후에 사용한다
    # 재귀 함수이므로 x 가 마지막 호출이 아닌경우는  반복호출
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
        print(y)
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

# 전체로 뽑은 문단을 이제 그 문단의 줄 갯수에 맞게 text.wrap 이라는 라이브러리를 활용하여 분산시켜서 list 로 나눠준다.
def split_text(tranlated_texts, change_start_index, len_index):
    # 만약 문단이없는경우는 pass 하기
    if change_start_index == []:
        pass
    else:
        s = 0
        # 문단이 시작되는 부분을 보고, 만약 시작되는 부분이면 , text.wrap 을 이용하여, 그 시작되는 문단이 몇줄인지 확인하고
        # 그 문단의 줄만큼 통번역한 부분을 분배(insert)해준다
        for idx, x in enumerate(change_start_index):
            line = round (len(tranlated_texts[x+s]) / len_index[idx])

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

        # print('bbox_hi[idx]-15', type(bbox_hi[idx] - 15))
        #
        # print('title_font', title_font)
        start_x = ((bbox[0][0] + bbox[1][0]) // 2 - wi / 2)
        start_y = ((bbox[0][1] + bbox[2][1]) // 2 - hi / 2)
        image_editable.text((bbox[0][0], bbox[0][1]), text, color, anchor=None, font=title_font)

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
    bbox_list, text_list = array_box(bbox_list, text_list)
    print(text_list)
    print(' len - bbox_list : ', len(bbox_list))
    print(' len - text_list: ', len(text_list))
    bbox_list, text_list = sum_box(bbox_list, text_list, x)

    print('change len - bbox_list : ', len(bbox_list))
    print(' len - text_list: ', len(text_list))
    print(text_list)
    text_list, change_start_index, len_index = jaegi(bbox_list, text_list, x, z)
    print(change_start_index, len_index)
    tranlated_texts: List[str] = translate_texts(texts=text_list, type='naver')
    print(f'Tranlated_texts : {tranlated_texts}')
    tranlated_texts = split_text(tranlated_texts, change_start_index, len_index)
    print(tranlated_texts)
    print(tranlated_texts)

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
        # img = change_original(img,img_cut, bbox)
    #
    save_inpainting_images()

    rewrite(tranlated_texts,bbox_list,color_list)

