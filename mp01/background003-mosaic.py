import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# 定義加入文字函式
def putText(x,y,text,size=70,color=(0,0,0)):
    global img
    fontpath = 'NotoSansTC-Regular.otf'            # 字型
    font = ImageFont.truetype(fontpath, size)      # 定義字型與文字大小
    imgPil = Image.fromarray(img)                  # 轉換成 PIL 影像物件
    draw = ImageDraw.Draw(imgPil)                  # 定義繪圖物件
    draw.text((x, y), text, fill=color, font=font) # 加入文字
    print("x:{},y:{},text:{}".format(x,y,text))
    img = np.array(imgPil)                         # 轉換成 np.array


if __name__ == '__main__':
    img = cv2.imread('1.png')
    #img = np.zeros((300, 300, 3), dtype='uint8')
    size = img.shape         # 取得原始圖片的資訊
    h, w, c = img.shape

    x = w-(w*0.6)    # 剪裁區域左上 x 座標
    y = h-(h*0.7)    # 剪裁區域左上 y 座標

    print(x,y)
    #cv2.rectangle(img, (x, y), (x + 10, y + 10), (0, 0, 255), 5)
    #cv2.rectangle(img, (50,50), (250,250), (0, 0, 255), 5)
    putText(x,y,'台科大')     # 放入文字
    cv2.imshow('M10609906', img)

    while True:
        keyb = cv2.waitKey(1) & 0xFF
        if keyb == 27:
            break
        elif keyb == ord('0'):
            bg_image = None
        elif keyb == ord('1'):
            level = 10  # 縮小比例 ( 可當作馬賽克的等級 )
            h = int(size[0] / level)  # 按照比例縮小後的高度 ( 使用 int 去除小數點 )
            w = int(size[1] / level)  # 按照比例縮小後的寬度 ( 使用 int 去除小數點 )
            mosaic = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)  # 根據縮小尺寸縮小
            mosaic = cv2.resize(mosaic, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)  # 放大到原本的大小
            cv2.imshow('oxxostudio', mosaic)

        elif keyb == ord('2'):
            x = 500  # 剪裁區域左上 x 座標
            y = 450  # 剪裁區域左上 y 座標
            cw = 570  # 剪裁區域寬度
            ch = 120  # 剪裁區域高度

            #使用陣列切片裁剪圖片
            crop_img = img[y:y+ch, x:x+cw]  # 取出陣列的範圍
            cv2.imwrite('img cut.jpg', crop_img)  # 儲存圖片
            cv2.imshow('M10609906-1', crop_img)

            #cv2.rectangle(img, (x,y),(x+cw, y+ch),(0,0,255),1)
            mosaic2 = img[y:y + ch, x:x + cw]  # 取得剪裁區域
            level = 15  # 馬賽克程度
            h = int(ch / level)  # 縮小的高度 ( 使用 int 去除小數點 )
            w = int(cw / level)  # 縮小的寬度 ( 使用 int 去除小數點 )
            mosaic2 = cv2.resize(mosaic2, (w, h), interpolation=cv2.INTER_LINEAR)
            mosaic2 = cv2.resize(mosaic2, (cw, ch), interpolation=cv2.INTER_NEAREST)
            img[y:y + ch, x:x + cw] = mosaic2  # 將圖片的剪裁區域，換成馬賽克的圖
            cv2.imshow('M10609906', img)

    # 按下任意鍵停止
    cv2.destroyAllWindows()