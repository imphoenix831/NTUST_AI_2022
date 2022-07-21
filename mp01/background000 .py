import cv2
import mediapipe as mp
import numpy as np
import time
prev_time = 0
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(0)

cap = cv2.VideoCapture(0)
while(True):
    success, frame = cap.read()

    h, w, c = frame.shape

    print(frame.shape)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bg_image0 = np.zeros(image.shape, dtype=np.uint8)
    bg_image0[:] = (255, 255, 0)
    results = selfie_segmentation.process(image)

    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

    output_image = np.where(condition, frame, bg_image0)

    cv2.putText(output_image, f'FPS: {int(1 / (time.time() - prev_time))}',
                (3, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    prev_time = time.time()

    cv2.imshow('MediaPipe Selfie Segmentation 0', output_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()