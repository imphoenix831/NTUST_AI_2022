import cv2
import face_recognition
import os
import numpy as np
path = '../picc' #已知人物相片存放的目錄
pic = 1
images = []
classNames = []
encodeListKnown=[]

myList = os.listdir(path)
print("myList=", myList)

print("把data目錄中的相片一一讀入")
for cl in myList:
    curImg = cv2.imdecode(np.fromfile(file=f'{path}/{cl}', dtype=np.uint8),
                          cv2.IMREAD_COLOR)
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])   # 把相片(人物)的名稱存到classNames中

print("把相片送去編碼，結果一一存到 encodeListKnown中")
for img in images:
    print("處理第"+str(pic)+"幾張相片"+ classNames[pic-1])
    pic +=1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encodeListKnown.append(encode)

print("把相片(人物)的名稱寫入classNamesFile.txt中  完成!!")
with open("classNamesFile.txt", "w") as f:
    for i in range(len(images)):
        f.write(classNames[i]+'\n')

print("把相片(人物)的編碼寫入array.bin中  完成!!")
with open("array.bin", "wb") as f:
    for i in range(len(encodeListKnown)):
        encodeListKnown[i].tofile(f)

# print(encodeListKnown)
# print('classNames = ', classNames)
# print('# of imgs = ', len(images))
# print('people in encodeListKnown = ', len(encodeListKnown))
print('全部完成！！！！！！')