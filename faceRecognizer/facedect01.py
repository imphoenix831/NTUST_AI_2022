import cv2
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image


#detector = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')  # 載入人臉追蹤模型

detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')  # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列


# 定義加入文字函式
def putText(image, x, y, text, size=30, color=(0, 0, 0)):
    fontpath = 'font/NotoSansTC-Regular.otf'  # 字型
    font = ImageFont.truetype(fontpath, size)  # 定義字型與文字大小
    imgPil = Image.fromarray(image)  # 轉換成 PIL 影像物件
    draw = ImageDraw.Draw(imgPil)  # 定義繪圖物件
    text = '圖片'+ str(text)
    draw.text((x, y), text, fill=color, font=font)  # 加入文字
    image = np.array(imgPil)  # 轉換成 np.array
    return image

def image_show(id, image_len ):
    for i in range(1, image_len):
        img = cv2.imread(f'detect/{id}/{i}.jpg')  # 依序開啟每一張蔡英文的照片
        img = cv2.resize(img, (640, 480))

        img = putText(img, 100, 10, str(i))  # 放入文字
        cv2.imshow('face', img)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:
            print('break')
            break
        elif key == 32:
            continue
        else:
            time.sleep(5)


def image_detect( id , image_len , name=''):
    global faces
    global ids

    id1= int(id)
    for i in range(1, image_len):
        img = cv2.imread(f'detect/{id}/{i}.jpg')  # 依序開啟每一張蔡英文的照片
        img = cv2.resize(img, (640, 480))

        winame ='face_'+ id

        cv2.namedWindow(winame)
        cv2.moveWindow(winame, (id1*650), 30)  # Move it to (40,30)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
        img_np = np.array(gray, 'uint8')  # 轉換成指定編碼的 numpy 陣列

        face = detector.detectMultiScale(gray, scaleFactor=1.1 , minNeighbors=3)  # 擷取人臉區域

        for (x, y, w, h) in face:
            cv2.rectangle(img, (x,y) , (x+w, y+h), (0,255,0), 1 )
            faces.append(img_np[y:y + h, x:x + w])  # 記錄蔡英文人臉的位置和大小內像素的數值
            ids.append(id1)  # 記錄蔡英文人臉對應的 id，只能是整數，都是 1 表示蔡英文的 id 為 1

        img = putText(img, 100, 10, str(i))  # 放入文字
        cv2.imshow(winame, img)

        if cv2.waitKey(100) == ord('q'):  # 每一毫秒更新一次，直到按下 q 結束
            break

    return img

def camera_detect( id , name='' ):
    global faces
    global ids
    print('camera...')  # 提示啟用相機
    id1 = int(id)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
        img_np = np.array(gray, 'uint8')  # 轉換成指定編碼的 numpy 陣列
        face = detector.detectMultiScale(gray)  # 擷取人臉區域

        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            faces.append(img_np[y:y + h, x:x + w])  # 記錄自己人臉的位置和大小內像素的數值
            ids.append(id1)  # 記錄自己人臉對應的 id，只能是整數，都是 1 表示川普的 id

        cv2.imshow('Camera', img)  # 顯示攝影機畫面

        if cv2.waitKey(100) == ord('q'):  # 每一毫秒更新一次，直到按下 q 結束
            break

    cap.release()


if __name__ == '__main__':

    #image_show('01', 16)
    #image_show('02', 16)
    #img1 = image_detect('01', 16)
    #img2 = image_detect('02', 16)

    camera_detect('03')

    # concatenate image Horizontally
    Hori = np.concatenate((img1, img2), axis=1)

    # concatenate image Vertically
    Verti = np.concatenate((img1, img2), axis=0)

    cv2.imshow('HORIZONTAL', Hori)
    #cv2.imshow('VERTICAL', Verti)

    print(ids)
    print(faces)

    print('training...')  # 提示開始訓練
    recog.train(faces, np.array(ids))  # 開始訓練
    recog.save('face.yml')  # 訓練完成儲存為 face.yml
    print('ok!')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
