import cv2

# 加載模型
#net = cv2.dnn.readNetFromTorch('../models/starry_night.t7')  # style image : 梵谷
net = cv2.dnn.readNetFromTorch('../models/mosaic.t7')  # style image : 梵谷
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV);
# 讀取圖片
image = cv2.imread('../../data/IMG_2997.png')    # content.jpg :
cv2.imshow('Original image', image)

(h, w) = image.shape[:2]

#blog: 從圖檔,轉成 Blog
#在圖片進行訓練前，都要將圖像減去imagenet的均值，為什麼呢？
#去均值是為了對圖像進行標準化，可以移除圖像的平均亮度值。
#很多情況下我們對圖像的照度並不感興趣，而更多地關注其內容，圖像的整體明亮程度並不會影響圖像中存在的是什麼物體
#在每個樣本上減去數據的統計平均值可以移除共同的部分，凸顯個體差異。其效果如下所示

blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
# 進行計算

# call DNN Model
net.setInput(blob)
out = net.forward()
out = out.reshape(3, out.shape[2], out.shape[3])

#把均值加回
out[0] += 103.939
out[1] += 116.779
out[2] += 123.68
out /= 255
out = out.transpose(1, 2, 0)

# 輸出圖片
cv2.imshow('Styled image', out)
cv2.waitKey(0)
