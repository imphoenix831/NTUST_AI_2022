import cv2

#cap = cv2.VideoCapture(0)

#可以讀 mp4 中的檔案
cap = cv2.VideoCapture("background.mp4")

while(True):
    success, frame = cap.read()

    cv2.imshow('AI000', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()