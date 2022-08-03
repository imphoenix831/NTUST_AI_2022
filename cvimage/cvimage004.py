import cv2
import numpy as np

img1=cv2.imread("data/1.png")
img1= cv2.resize(img1, (320,200))

img2=cv2.imread("data/2.png")
img2= cv2.resize(img2, (320,200))

#add 二張圖的 size 要相同
img3 = cv2.add(img1, img2)

img4 = cv2.addWeighted(img1, 0.7,img2, 0.3, 0)

imghor = np.hstack((img1,img2,img3, img4))
imgv = np.vstack((img1,img2,img3, img4))

cv2.imshow("Image1",imghor)
cv2.imshow("Image2",imgv)

cv2.waitKey(0)
