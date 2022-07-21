import cv2
import mediapipe as mp
import numpy as np
import time
prev_time = 0
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(0)
bg_image1 =cv2.imread('1.png')

cap = cv2.VideoCapture(0)
while (True):
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    h, w, c = frame.shape
    print(frame.shape)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bg_image1 = cv2.resize(bg_image1, (w, h))

    results = selfie_segmentation.process(image)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    output_image = np.where(condition, frame, bg_image1)
    cv2.putText(output_image, f'FPS: {int(1 / (time.time() - prev_time))}',
                (3, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    prev_time = time.time()
    cv2.imshow("Selfie Segmentation 1", output_image)
    keyb = cv2.waitKey(1) & 0xFF
    if  keyb == 27:
        break
cap.release()
cv2.destroyAllWindows()