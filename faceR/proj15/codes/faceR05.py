import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# encodeListKnown = []
with open("classNamesFile.txt", "r") as f:  # 從classNamesFile.txt中讀出相片(人物)的名稱
    classNames= f.read().splitlines()
print('classNames = ', classNames)

with open("array.bin", "rb") as f:  # 從array.bin中讀出相片的編碼
    encodeListKnown=np.fromfile(f)
encodeListKnown= encodeListKnown.reshape(-1, 128)

print('讀入 ', len(encodeListKnown),'個人物編碼 ')
image = cv2.imread("../data/pin3.jpg")  # 用cv2.imread去讀檔案，記得cv2讀入檔案是 BGR 的型態
facesCurFrame = face_recognition.face_locations(image)# , model="cnn")  # 讀到的圖片找出臉的座標
encodesCurFrame = face_recognition.face_encodings(image, facesCurFrame)  # 把座標送去編碼

for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):  # 將座標與編碼兩個list合起來處理
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # 比對成功matches[i]=True
    print('matches = ', matches)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    print("faceDis = ", faceDis)
    matchIndex = np.argmin(faceDis)
    if matches[matchIndex]:  # 如果距離最短的臉孔，在已知臉孔編碼比對為True時
        name = classNames[matchIndex]
    else:  # 如果編碼比對為False時，代表不在名單中
        name = '不在名單中'
    print("matchIndex = ", matchIndex, name)
    y1, x2, y2, x1 = faceLoc
    y1, x2, y2, x1 = y1 , x2 , y2 , x1
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.rectangle(image, (x1, y2 + 22), (x2, y2), (0, 0, 255), cv2.FILLED)
    font = ImageFont.truetype("simsun.ttc", 16)  # 導入中文字型
    img_pil = Image.fromarray(image)  # 將numpy array 的圖片格式轉為PIL 的圖片格式
    draw = ImageDraw.Draw(img_pil)  # 創建畫板
    draw.text((x1 + 8, y2 + 2), name, font=font, fill=(255, 255, 255, 1))  # 在圖片上畫中文
    image = np.array(img_pil)

cv2.imshow('MATCH', image)
cv2.waitKey(0)
