import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_russian_plate_number.xml")

# cap = cv2.VideoCapture('lp2.mp4')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in nplate:
        wT, hT, cT = img.shape
        a, b = (int(0.02 * wT), int(0.02 * hT))
        plate = img[y + a:y + h - a, x + b:x + w - b, :]
        # make the img more darker to identify LPR
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        stat = read[0:2]
        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x - 1, y - 40), (x + w + 1, y), (51, 51, 255), -1)
        cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Result", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
