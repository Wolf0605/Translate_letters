# Translate_letters
## Goal
<img src='https://user-images.githubusercontent.com/82213429/131771696-e8e07453-f140-456d-8489-4388b0522ad5.jpg' width='500' >

<img src='https://user-images.githubusercontent.com/82213429/131773029-30fff46c-d016-4f12-a225-5a06af8f40fb.png' width='500'>
<img src='https://user-images.githubusercontent.com/82213429/131773179-b89b4ae0-22de-4f82-a8fd-4a11703b37ac.png' width='500'>

<img src='https://user-images.githubusercontent.com/82213429/131773266-9cc048de-4161-4d0f-b06f-d172c4346e4d.png' width='500'>
<img src='https://user-images.githubusercontent.com/82213429/131773269-39c1f0e6-b8ef-4ebd-b985-52d30fae52b0.png' width='500'>

Helpful link : https://www.youtube.com/watch?v=lOQTiDY-gbA
## How can we do this
### 1. Find letters and shows the box  
<img src='https://user-images.githubusercontent.com/82213429/131785370-85f06d28-41f0-4db0-85db-46dd316433a0.png' width='500'>

### 2. Show and translate letteres that in boxes
 
NETFLEX     ->     넷플렉스

### 3. Find the color which maxium similiar with background and then cover them
<img src='https://user-images.githubusercontent.com/82213429/131785447-d76aa2f8-407e-44df-bd7f-da76b70d9a06.png' width='500'>

### 4.  And then rewrite the letter that transalte to korean as similiary as we can
<img src='https://user-images.githubusercontent.com/82213429/131785502-cb55dfef-0d31-4842-a960-92ebff27b911.png' width='500'>


## Require technic
### 1.OCR
1. 직접 ocr 구현
2. Pytorch or tesseract  에서 ocr 라이브러리 가져오기, easy ocr 라이브러리 사용
3. Naver clova ocr, google ocr API 구매

### 2.Translate
1.PaPago or Google translate API 가져오기

+a ) ocr 에서 이상하게 출력된(정확도가 낮은) text 를 멀쩡하게 바꿔주는 작업 (LSTM, RNN )

### 3. Inpainting (난이도 높을예정)
1. ocr 에 나온 각들을 주워서 블러핑 사용
2. 글자들의 형태를 따서 주변색이랑 매칭


### 4.  Adobe photoshop, Adobe typekit

<img src='https://user-images.githubusercontent.com/82213429/135216131-d46968f8-5de1-4c8d-a1dd-a67c80c70359.png' width ='300'>

mijung

## Limit ( 제한할 사항 )
비디오란 이미지를 이어 붙힌건데 글자가 많은 사진은 컴퓨터가 해석하는데 길게는 10초도 걸린다.
1.이걸 어떻게 해서 시간을 줄일것 인가??

2.비디오에는 수많은 장면이 있는데, 나오는 글자마다 전부 해석할 것인가?.
NO, 테이크가 긴곳만 할예정
**예시**

![200](https://user-images.githubusercontent.com/82213429/135221823-84cf1752-450e-4ef6-b167-bbe790537e33.gif)
![1ST0](https://user-images.githubusercontent.com/82213429/135223127-2e38f9b8-675c-42c8-9c89-243de57f55cf.gif)

**아닌예시**

![211155](https://user-images.githubusercontent.com/82213429/135223186-70e215a8-61af-4195-81c5-bd14e8dbf1da.gif)
![200 (1)](https://user-images.githubusercontent.com/82213429/135223176-55032d6d-a2c6-41da-9c90-988f45696b0c.gif)
