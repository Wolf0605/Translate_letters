import cv2
import matplotlib.pyplot as plt
import easyocr
import sys
import googletrans
from typing import List
import requests
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# 이미지 파일 경로
file_path = r'sign_noun_002_33753.jpg'
img = cv2.imread(file_path, cv2.IMREAD_COLOR)

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
    return np.array(bbox_list), text_list


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
def cut_image(img, bbox):
    x_min = bbox[0, 0]
    x_max = bbox[1, 0]
    y_min = bbox[0, 1]
    y_max = bbox[2, 1]

    img = img[y_min:y_max, x_min:x_max]

    return img

def mask_image(img2):
    # masking 작업
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

    # 글씨 색이 밝든 어둡든 masking 씌워주기
    return_rgb = rgb(img2)
    if return_rgb == 0:
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # plt.imshow(mask)
    # plt.show()
    return mask


def change_original(masked_img, bbox):
    x_min = bbox[0, 0]
    x_max = bbox[1, 0]
    y_min = bbox[0, 1]
    y_max = bbox[2, 1]

    img[y_min:y_max, x_min:x_max] =  masked_img
    return img

def rgb(img):
    r1, g1, b1 = img[0][0]
    r2, g2, b2 = img[-1][0]
    r3, g3, b3 = img[-1][-1]
    r4, g4, b4 = img[0][-1]
    # 배경이 밝은 부분이 한 부분이라도 있으면
    ##
    # 수정필요함 (귀퉁이 4개중 2개 이상이 흰색이면 이런식으로 )
    if (r1>=0 and g1>150 and b1>150) or (r2>150 and g2>=0 and b2>150)\
            or (r3>150 and g3>150 and b3>=0) or (r4>150 and g4>150 and b4>150):
        return 0

def rewrite(tranlated_texts ,bbox_list):
    # 폰트( 구글 폰트에서 이용가능 ) , 폰트크기
    image_path = 'output_inpainting'
    img = Image.open(f"{image_path}/{file_path}")

    image_editable = ImageDraw.Draw(img)

    # (x, y ) , ( 237, 230, 211) 색감
    for idx, bbox in enumerate(bbox_list):
        text = tranlated_texts[idx]
        title_font = ImageFont.truetype('ttf/NotoSansKR-Bold.otf', 1)
        wi, _ = title_font.getsize(text)
        # bbox_wi = bbox[1][0] - bbox[0][0]
        bbox_hi = bbox[2][1] - bbox[1][1]

        font_size = decsion_font_size(bbox_hi, text)
        title_font = ImageFont.truetype('ttf/NotoSansKR-Bold.otf', font_size)
        image_editable.text((bbox[0][0], bbox[0][1]), text, (255,255,255), anchor = 'lt', font=title_font)

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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bbox_list, text_list = easy_ocr_result(img)
    print('Text_list :', text_list)
    tranlated_texts: List[str] = translate_texts(texts=text_list, type='naver')
    print(f'Tranlated_texts : {tranlated_texts}')

    for bbox in bbox_list:
        img_cut = cut_image(img, bbox)
        # plt.imshow(img_cut)
        # plt.show()
        mask = mask_image(img_cut)
        masked_img = cv2.inpaint(img_cut, mask, 3, cv2.INPAINT_TELEA)
        img = change_original(masked_img, bbox)

    save_inpainting_images()
    rewrite(tranlated_texts,bbox_list)



