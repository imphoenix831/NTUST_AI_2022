import cv2
import face_recognition
import os

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img, model="cnn")[0]
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
