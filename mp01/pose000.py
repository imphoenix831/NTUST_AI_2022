import cv2
import mediapipe as mp

#conn = mp.solutions.pose.POSE_CONNECTIONS


'''
conn = {(5, 6), (24, 26), (26, 28), (11, 23), (25, 27), (15, 19), (3, 7), (27, 29), (1, 2), 
        (28, 30), (29, 31), (12, 24), (16, 22), (0, 4), (16, 18), (18, 20), (11, 13), (23, 25), (17, 19), (12, 14), 
        (4, 5), (13, 15), (9, 10), (2, 3), (28, 32), (27, 31), (11, 12), (23, 24), (15, 17), (0, 1), (6, 8), 
        (14, 16), (15, 21), (16, 20)}

'''
conn = {(5, 6), (3, 7),  (1, 2),  (0, 4), (4, 5),  (9, 10), (2, 3), (0, 1), (6, 8) }
print("Conn:{}".format(conn))

pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#把 pose 中的點和線畫出來
mp_drawing = mp.solutions.drawing_utils

#pose 中的點和線 的顏色和大小

#使用官方的點.線 style
#spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

#設定個人化的線
spec = mp_drawing.DrawingSpec(color=(255, 255, 0),thickness=3, circle_radius=1)



cap = cv2.VideoCapture(0)
while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    continue
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  results = pose.process(image)


  if results.pose_landmarks:
      #把 Pose 的點線, 依 conn, spec 畫出來
      mp_drawing.draw_landmarks(frame, results.pose_landmarks, conn, spec)

  cv2.imshow('MediaPipe Pose', frame)
  if cv2.waitKey(5) & 0xFF == 27:
    break
cap.release()