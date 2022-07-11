import cv2

# 選擇攝影機
cap = cv2.VideoCapture(2)

while(True):
  # 從攝影機擷取一張影像
  success, frame = cap.read()

  # 顯示圖片
  cv2.imshow('AI001', frame)

  # 若按下 esc 鍵則離開迴圈
  if cv2.waitKey(1) & 0xFF == 27:
    break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()