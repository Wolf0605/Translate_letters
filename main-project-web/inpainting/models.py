from django.db import models
import matplotlib.pyplot as plt
import PIL
from io import BytesIO
from django.core.files.base import ContentFile
import numpy as np 
import os
import cv2
from pathlib import Path
from TEST_JW.test import (
    easy_ocr_result,
    translate_texts,
    cut_image,
    mask_image,
    change_original,
    rewrite,
)


class Image(models.Model):
    image = models.ImageField(upload_to='image/',blank=True, null=True)

    def save(self, *args, **kwargs):
        # 이미지 열기

        img_pil = PIL.Image.open(self.image)
        bbox_list, text_list = easy_ocr_result(img_pil)
        tranlated_texts = translate_texts(texts=text_list, type='naver')

        # numpy img

        img_np = np.array(img_pil)
        for bbox in bbox_list:
            img_cut = cut_image(img_np, bbox)
        
            mask = mask_image(img_cut)
            masked_img = cv2.inpaint(img_cut, mask, 3, cv2.INPAINT_TELEA)
            img_np = change_original(img_np,masked_img, bbox)

        img_pil = PIL.Image.fromarray(img_np)
        # print('type',type(img_pil))
        img = rewrite(img_pil, tranlated_texts,bbox_list)
        #  convert back to pil image
        # im_pil = Image.fromarray(img)
        # plt.imshow(img)
        # plt.show()
        # 저장
        buffer = BytesIO()
        img.save(buffer, format='png')
        image_png = buffer.getvalue()

        self.image.save(str(self.image), ContentFile(image_png), save=False)

        super().save(*args,**kwargs)