import cv2
import pafy
import mediapipe as mp
import numpy as np

pose = mp.solutions.pose.Pose()
conn = mp.solutions.pose.POSE_CONNECTIONS
mp_drawing = mp.solutions.drawing_utils
spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

#url = "https://www.youtube.com/watch?v=Cw-Wt4xKD2s"
#url = "https://www.youtube.com/watch?v=l7_QO-UVZRA"
# url = "https://www.youtube.com/watch?v=zQx4kv-JDwc"
# url = "https://www.youtube.com/watch?v=5u5mXzSPS98"
# url = "https://www.youtube.com/watch?v=Kzmb38__9iY"
url = "https://www.youtube.com/watch?v=OB8CmODjLvI"
# url = "https://www.youtube.com/watch?v=81zXRHX6Q_U"

live = pafy.new(url)
stream = live.getbest(preftype="mp4")
cap = cv2.VideoCapture(stream.url)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    continue
  frame = cv2.resize(frame, (576, 324))

  #原影片中的畫面
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  #畫出黑畫面 , numpy 產生 3維陣列, 並填上 zero 0
  frame0 = np.zeros(image.shape, dtype=np.uint8)

  results = pose.process(image)
  if results.pose_landmarks:
      mp_drawing.draw_landmarks(frame, results.pose_landmarks, conn, spec)
      mp_drawing.draw_landmarks(frame0, results.pose_landmarks, conn, spec)
  cv2.imshow('MediaPipe Pose Video', frame)
  cv2.imshow('MediaPipe Pose Black', frame0)
  if cv2.waitKey(5) & 0xFF == 27:
    break
cap.release()