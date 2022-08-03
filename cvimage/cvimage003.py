import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8) #設定為黑底，全部為0的矩陣
img[:] = 255,0,0 #改為藍色

img[200:300,100:300]= 0,255,0  #中間一塊設為綠色

cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),1) #對角線
cv2.rectangle(img,(0,0),(255,350),(0,0,255),1) #紅色方塊
cv2.circle(img,(400,50),30,(255,255,0),5)
cv2.putText(img," NTUST ",(300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),2)

cv2.imshow("Image",img)
cv2.waitKey(0)
