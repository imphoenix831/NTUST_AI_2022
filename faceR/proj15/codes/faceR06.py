# faceR06.py

import cv2
import face_recognition
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import time
import codecs

keyb = 0
with open("classNamesFile.txt", "r") as f:  # 從classNamesFile.txt中讀出相片(人物)的名稱
    classNames= f.read().splitlines()
print('classNames = ', classNames)

with open("array.bin", "rb") as f:
    encodeListKnown=np.fromfile(f)
encodeListKnown= encodeListKnown.reshape(-1, 128)
print('讀入 ', len(encodeListKnown),'個人物編碼 ')

cap = cv2.VideoCapture(0)

while True:
    prev_time = time.time()
    success, frame = cap.read() #把影片中每一個frame讀出，放到image
    frame = cv2.resize(frame,(800, 450))
    imgS = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    # facesCurFrame = face_recognition.face_locations(frame, model="cnn")
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        print('matches = ', matches)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("faceDis = ", faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:  # 如果距離最短的臉孔，在已知臉孔編碼比對為True時
            name = classNames[matchIndex]
        else:     # 如果編碼比對為False時，代表不在名單中
            name = '不在名單中'
        print("matchIndex = ", matchIndex, name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
        # y1, x2, y2, x1 = y1 , x2 , y2 , x1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y2 + 25), (x2, y2), (0, 0, 255), cv2.FILLED)
        font = ImageFont.truetype("simsun.ttc", 20)  # 導入中文字型
        img_pil = Image.fromarray(frame)  # 將numpy array 的圖片格式轉為PIL 的圖片格式
        draw = ImageDraw.Draw(img_pil)  # 創建畫板
        draw.text((x1 + 6, y2 + 2), name, font=font, fill=(255, 255, 255, 1))  # 在圖片上畫中文
        frame = np.array(img_pil)
        with codecs.open('../data/report.txt', 'r+',encoding='utf-8') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            if (name not in nameList) or (keyb == ord('s')):
                f.writelines(f'\n{name},{dtString}')

    cv2.putText(frame, 'fps: ' + str(int(1 / (time.time() - prev_time))), (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('Webcam match', frame)
    keyb = cv2.waitKey(1) & 0xFF
    if keyb == 27:
        break