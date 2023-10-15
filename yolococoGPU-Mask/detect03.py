import torch
import numpy as np
import cv2
import time
import pafy

prev_time = 0
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#cap = cv2.VideoCapture("https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=13380")
#cap = cv2.VideoCapture("https://cctv.klcg.gov.tw/facd4662")
#cap = cv2.VideoCapture("https://cctv7.kctmc.nat.gov.tw/play/live.php?devid={79d8a119-9164-1f84-a1c8-da4f5fb99508}&L=dfe8885c1acbe4c5e5fb91e6d9bd3724")
#cap = cv2.VideoCapture("https://cctv1.kctmc.nat.gov.tw/f75bb280?t=0.9768836315531848")
#cap = cv2.VideoCapture("https://youtu.be/dY2cRNr5Buw")

#cap = cv2.VideoCapture("https://icam.tw/cam/NFB-CCTV-N1-S-94.9-M")

#阿里山雲海
url = "https://youtu.be/dY2cRNr5Buw"

#url = "https://youtu.be/WA7GEXVGAP0"
live = pafy.new(url)
stream = live.getbest(preftype="mp4")
cap = cv2.VideoCapture(stream.url)



while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    frame = cv2.resize(frame,(960,540))
    results = model(frame)
    output_image = np.squeeze(results.render())
    cv2.putText(output_image, f'FPS: {int(1 / (time.time() - prev_time))}',
                (3, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    prev_time = time.time()
    cv2.imshow('YOLO COCO 02', output_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()