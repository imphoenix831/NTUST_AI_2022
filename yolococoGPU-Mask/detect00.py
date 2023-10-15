import torch
import cv2
import numpy as np

# model: yolov5s : image 1/1: 534x800 5 persons
# model: yolov5m : image 1/1: 534x800 6 persons, 1 boat
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
# model.conf = 0.5
# print(model)
img=cv2.imread('data/IMG_2997.png')

#把 image 送到 model 去分析, 將結果儲存在 resutls
results = model(img)
results.print()   #image 1/1: 534x800 5 persons
print(results.xyxy) #畫出 object 的 x,y 軸

#results.render() , 1維, 再加上 x,y

#results.xyxy[0]  # img1 predictions (tensor)
#results.pandas().xyxy[0]  # img1 predictions (pandas)

cv2.imshow('YOLO COCO', np.squeeze(results.render()))
cv2.waitKey(0)


