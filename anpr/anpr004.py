import cv2
import numpy as np
import imutils
import easyocr

img = cv2.imread('image1.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)   # Noise reduction
edged = cv2.Canny(bfilter, 30, 200)               # Edge detection
imghor = np.hstack((gray, bfilter, edged))
cv2.imshow("imghor_image", imghor)
keypoints = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find Contours
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
mask = np.zeros(gray.shape, np.uint8)  #black mask
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    all_image = cv2.drawContours(mask, [approx], 0, 255, -1)
    if len(approx) == 4:   # find the rectangle contour
        location = approx
        break
cv2.imshow("all_contours", all_image)
mask = np.zeros(gray.shape, np.uint8)
try:
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("new_image", new_image)
    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    reader = easyocr.Reader(['en'])
    text = reader.readtext(cropped_image)[0][-2]
    res = cv2.putText(img, text,(approx[0][0][0], approx[1][0][1]+60), 3, 1, (0,255,0), 2, cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
    cv2.imshow("found!", res)
except:
    print('error')
    res = cv2.putText(img, 'error', (130, 240), 3, 4, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.imshow("error", img)
cv2.waitKey(0)