import cv2
import easyocr

img = cv2.imread('image5.png')

#img 轉灰階
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#去雜訊 ,類美肌
#cv2.bilateralFilter(image, bi_kisze, sigma, sigma)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
#邊緣化
#cv2.canny(bfilter,min_threshole, max_threshold2)
canny_edge = cv2.Canny(bfilter,120, 200) #Canny Edge Detection
# Find contours based on Edges
#cv2.findContours : 找出圖形中的方框 , 多邊形, 三角形
#cv2.findContours(image, cv2_RETR_TREE:建立一個等級樹結構的輪廓,cv2.CHAIN_APPROX_SIMPLE 壓縮水平方向，垂直方向，對角線方向的元素，只保留該方向的終點座標，矩形輪廓只需4個點來儲存輪廓資訊

contours, new = cv2.findContours(canny_edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours=sorted(contours, key = cv2.contourArea, reverse = True)[:10]
for contour in contours: # Find the closed contour with 4 potential corners
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        #找出車牌的四邊形 len(approx) == 4
        if len(approx) == 4: #see whether it is a Rect
            x, y, w, h = cv2.boundingRect(contour)
            license_plate = gray[y:y + h, x:x + w]
            break

#easyocr 用英文; 中文要去 download
reader = easyocr.Reader(['en'])
result = reader.readtext(license_plate)

text = result[0][-2]
image = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),3)

image = cv2.putText(img, text, (x-30,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,250,0), 2, cv2.LINE_AA)

cv2.imshow("License Plate Detection",cv2.resize(image,(400,300)))
cv2.waitKey(0)