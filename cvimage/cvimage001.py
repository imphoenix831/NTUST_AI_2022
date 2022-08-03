import cv2
import numpy as np

img =cv2.imread("data/5.png")

cv2.resize(img, (32,20))
print('img.shap = ', img.shape)

#變成灰階
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('imgGray.shap = ', imgGray.shape)

#高斯模糊
imgBlur = cv2.GaussianBlur(imgGray,(5,5),0)
print('imgBlur.shap = ', imgBlur.shape)

#外框
imgCanny = cv2.Canny(imgGray, 300,300)
print('imgCanny.shap = ', imgCanny.shape)

imgCanny3 = cv2.Canny(img, 300,300)
print('imgCanny3.shap = ', imgCanny3.shape)

cv2.imshow("Output gray", imgGray)
cv2.imshow("Output blur", imgBlur)
cv2.imshow("Output canny", imgCanny)
cv2.imshow("Output canny3", imgCanny3)

imgGray3 = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)
imghor3 = np.hstack((imgGray3,img))
cv2.imshow("Output 3", imghor3)

imghor = np.hstack((imgGray,imgBlur,imgCanny))
imgver = np.vstack((imghor,imghor))
cv2.imshow("Output all", imgver)


cv2.waitKey(0) #代表無限等待，不然一下就消失，1代表0.001秒