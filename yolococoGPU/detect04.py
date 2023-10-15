# detect03.py
import numpy as np
import cv2
import torch
import pafy

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/best.pt',force_reload=True)
cap = cv2.VideoCapture(0)

# 台灣櫻花鉤吻鮭實境秀
#url = "https://www.youtube.com/watch?v=BzbvgjpDNdE"
#live = pafy.new(url)
#stream = live.getbest(preftype="mp4")
#cap = cv2.VideoCapture(stream.url)


while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    frame = cv2.resize(frame,(800,480)) #BGR

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)  #RGB
    # print(np.array(results.render()).shape)

    frame2 = cv2.cvtColor(np.squeeze(results.render()), cv2.COLOR_RGB2BGR)
    cv2.imshow('YOLO COCO 03 mask detection', frame2)

    #cv2.imshow('YOLO COCO 03 mask detection', np.squeeze(results.render()))


    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
