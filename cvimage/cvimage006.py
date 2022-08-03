import cv2
import numpy as np

from PIL import ImageFont, ImageDraw, Image

def nothing(x):
    pass

font = ImageFont.truetype("simsun.ttc", 20) #導入中文字型

# imgtypes = ["original","HSV","Mask","result"]
imgtypesC = ["原始","HSV","遮罩","結果"]

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,300)

# 建立6條拉bar

cv2.createTrackbar("Hue Min 色相","TrackBars",0,179,nothing)  # 0: default value ; 179: max values
cv2.createTrackbar("Hue Max 色相","TrackBars",179,179,nothing)
cv2.createTrackbar("Sat Min 飽和","TrackBars",0,255,nothing)
cv2.createTrackbar("Sat Max 飽和","TrackBars",255,255,nothing)
cv2.createTrackbar("Val Min 亮度","TrackBars",0,255,nothing)
cv2.createTrackbar("Val Max 亮度","TrackBars",255,255,nothing)

while (True):
    img = cv2.imread("data/4.png")
    img = cv2.resize(img, (280, 200))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min 色相", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max 色相", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min 飽和", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max 飽和", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min 亮度", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max 亮度", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    # 用拉bar獨到的lower,upper來建立mask影像
    mask = cv2.inRange(imgHSV, lower, upper)

    #cv2.imshow("hsv", imgHSV)
    #cv2.imshow("mask",mask)
    # 把mask與原始影像用bitwise_and去背
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    imgmask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    imghor = np.hstack((img, imgHSV,imgmask, imgResult))

    # for i in range(0, len(imgtypesC)):
    #     cv2.putText(imghor, imgtypesC[i], ((10 + 280 * i), 20), 4, 0.6, (100, 10, 255), 1)

    # 中文輸出要必須將圖片轉成PIL格式，再用Draw的方式來畫出中文，畫好之後，再把PIL格式轉回numpy array
    img_pil = Image.fromarray(imghor)  # 將numpy array的圖片格式轉為PIL的圖片
    draw = ImageDraw.Draw(img_pil)  # 創建畫板
    for i in range(0, len(imgtypesC)):
        draw.text(((10 + 280 * i), 20), imgtypesC[i], font=font, fill=(100, 10, 255, 1))  # 在圖片上畫中文
    imghor = np.array(img_pil)  # 將PIL圖片轉回numpy array的格式

    cv2.imshow("all images", imghor)
    #cv2.imshow("TrackBars", imghor)

    if cv2.waitKey(5) & 0xFF == 27:
        break

