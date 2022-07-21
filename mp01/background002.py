import cv2
import mediapipe as mp
import numpy as np
import time

blur=0
prev_time = 0
bg_image = None
#BG_COLOR = (255, 255, 0)
BG_COLOR = (0, 0, 0)
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(0)

cap = cv2.VideoCapture(0)

#while (True):
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    h, w, d = frame.shape
    print(frame.shape)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(image)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.6

    if bg_image is None:
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
    else:
        bg_image = cv2.resize(bg_image, (w, h))
        if blur > 0:
            bg_image = cv2.GaussianBlur(bg_image, (55, 55), 0)
            blur = 0
    output_image = np.where(condition, frame, bg_image)

    cv2.putText(output_image, f'FPS: {int(1 / (time.time() - prev_time))}'
                ,(3, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    prev_time = time.time()
    cv2.imshow("Selfie Segmentation 2", output_image)

    keyb = cv2.waitKey(1) & 0xFF

    if  keyb == 27:
        break
    elif keyb == ord('0'):
        bg_image = None
    elif keyb == ord('1'):
        bg_image = cv2.imread('1.png')
    elif keyb == ord('2'):
        bg_image = cv2.imread('2.png')
    elif keyb == ord('3'):
        bg_image = cv2.imread('3.png')
    elif keyb == ord('b'):
        blur +=1
    elif keyb == ord('v'):
        blur -=1

cap.release()
cv2.destroyAllWindows()