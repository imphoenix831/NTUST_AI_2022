import cv2
import face_recognition
import os
import numpy as np

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

path = '../pic'
images = []
classNames = []
myList = os.listdir(path)
print("myList=", myList)

for cl in myList: #把已知人物相片目錄中的影像一一存到images這個list
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print('classNames = ', classNames)

encodeListKnown = findEncodings(images) #把images這個list中的相片送去編碼

print('people in encodeListKnown = ', len(encodeListKnown))
print('Encoding Complete')

image = cv2.imread("../data/pin3.jpg")  # 用cv2.imread去讀檔案，記得cv2讀入檔案是 BGR 的型態

facesCurFrame = face_recognition.face_locations(image)  # 讀到的圖片找出臉的座標
encodesCurFrame = face_recognition.face_encodings(image, facesCurFrame)  # 把座標送去編碼

for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):  # 將座標與編碼兩個list合起來處理
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # 比對成功matches[i]=True
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # 找出比對後的faceDis距離
    matchIndex = np.argmin(faceDis)  # 找出faceDis距離最小的Index

    if matches[matchIndex]:  # 如果這個有match時
        name = classNames[matchIndex].upper()  # 用這個index找出姓名後都改為大寫
        print("faceDis = ", faceDis)
        print("matchIndex = ", matchIndex)
        print(name)

        y1, x2, y2, x1 = faceLoc
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(image, (x1, y2 + 25), (x2, y2), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, name, (x1 + 6, y2 + 16), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

cv2.imshow('MATCH', image)
cv2.waitKey(0)
