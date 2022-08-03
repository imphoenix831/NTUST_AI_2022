import cv2
print("cv2 imported")

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("data/cck.mp4")
# cap = cv2.VideoCapture("https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=13380")

while cap.isOpened():
    if cv2.waitKey(1) & 0xFF == 27:
        break

    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    cv2.imshow("video", frame)


cv2.imshow("Output", img)
cap.release()
cv2.destroyAllWindows()