import cv2


img = cv2.imread("data/5.png")
img = cv2.resize(img,(800,480))


imgCropped = img[40:270, 260:560]   #img[y1:y2 , x1:x2]
print(img.shape)
print(imgCropped.shape)
cv2.imshow("Image",img)
cv2.imshow("Image Cropped",imgCropped)
cv2.waitKey(0)