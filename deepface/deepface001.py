#年齡、性別、人種偵測
import cv2
from deepface import DeepFace
import numpy as np

img = cv2.imread('test01.jpg')     # 讀取圖片

try:
    #analyze = DeepFace.analyze(img)  # 辨識圖片人臉資訊
    emotion = DeepFace.analyze(img, actions=['emotion'])  # emotion
    age = DeepFace.analyze(img, actions=['age'])  # age
    race = DeepFace.analyze(img, actions=['race'])  # 人種
    gender = DeepFace.analyze(img, actions=['gender'])  # gender


    print('emotion:{}'.format(emotion['dominant_emotion']))
    print(age['age'])
    print(race['dominant_race'])
    print(gender['gender'])

except:
    pass

cv2.imshow('M10609906', img)
cv2.waitKey(0)
cv2.destroyAllWindows()