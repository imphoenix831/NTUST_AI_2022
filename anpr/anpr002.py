import cv2
import easyocr
import numpy as np
img = cv2.imread('image5.PNG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
canny_edge = cv2.Canny(bfilter,120, 200)        #Canny Edge Detection
contours, new  = cv2.findContours(canny_edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours, key = cv2.contourArea, reverse = True)[:10]
horImg = np.hstack((gray, bfilter, canny_edge))
cv2.imshow("gray bfilter canny",cv2.resize(horImg,(1100,300)))
for contour in contours: # Find the closed contour with 4 potential corners
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:     #see whether it is a Rect
            x, y, w, h = cv2.boundingRect(contour)
            # license_plate = gray[y:y + h, x:x + w]
            license_plate = bfilter[y:y + h, x:x + w]
            break
try:
    reader = easyocr.Reader(['en'])
    result = reader.readtext(license_plate)
    text = result[0][-2]
    image = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),3)
    image = cv2.putText(img, text, (x-30,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,250,0), 2, cv2.LINE_AA)
    cv2.imshow("License Plate Detection",cv2.resize(image,(400,300)))
except:
    print('error')
    res = cv2.putText(img, 'error', (130, 240), 3, 4, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.imshow("error", img)
    # pass
cv2.waitKey(0)