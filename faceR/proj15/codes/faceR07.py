# faceR03.py-1  先建立編碼與紀錄兩個功能函式
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import mediapipe as mp
import streamlit as st
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec1 = mp_drawing.DrawingSpec(color=(0, 255, 0),thickness=1, circle_radius=1)
drawing_spec2 = mp_drawing.DrawingSpec(color=(0, 255, 255),thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def findEncodings(images): #把相片中的臉部編碼，存到encodeList
    encodeList = []
    try:
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img, model="cnn")[0]
            encodeList.append(encode)
    except:
        pass
    return encodeList

def markAttendance(name): #比對到的人，看看簽到檔中是否已經有名字，沒有的話就記錄姓名與時間
    with open('../data/report.txt','r+', encoding="utf-8") as f:

        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        if (name not in nameList):# or (keyb == ord('s')):
            f.writelines(f'\n{name},{dtString}')

path = '../pic' #已知人物相片存放的目錄
images = []
classNames = []
myList = os.listdir(path)

for cl in myList: # 把已知人物相片目錄中的影像一一存到images這個list
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

encodeListKnown = findEncodings(images) #把images這個list中的相片送去編碼
print('Encoding Complete')

cap = cv2.VideoCapture(0)  #用cv2.imread去讀檔案
st.title('Streamlit + CV2')
run = st.checkbox('請打勾來執行')
FRAME_WINDOW = st.image([])

while run:
    success, frame = cap.read() #把影片中每一個frame讀出，放到image
    facesCurFrame = face_recognition.face_locations(frame)
    encodesCurFrame = face_recognition.face_encodings(frame, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]: #如果這個Index有match 為True時
            name = classNames[matchIndex].upper() #姓名都改為大寫
            print("faceDis = ", faceDis)
            print("matchIndex = ", matchIndex)
            print(name)
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y2 + 25), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 + 16), 3, 0.6, (255, 255, 255), 1)
            markAttendance(name)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                      None, mp_drawing_styles.get_default_face_mesh_contours_style())
    FRAME_WINDOW.image(frame, channels='BGR')
    #cv2.imshow('Webcam match', frame)
    keyb = cv2.waitKey(100) & 0xFF
    if keyb == 27:
        break