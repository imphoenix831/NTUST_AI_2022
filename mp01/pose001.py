import cv2
import pafy
import mediapipe as mp

pose = mp.solutions.pose.Pose()
conn = mp.solutions.pose.POSE_CONNECTIONS
mp_drawing = mp.solutions.drawing_utils
spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

url = "https://www.youtube.com/watch?v=Z1u5wYKZ9DQ"
live = pafy.new(url)
stream = live.getbest(preftype="mp4")
cap = cv2.VideoCapture(stream.url)

while cap.isOpened():
  success, frame = cap.read()
  frame = cv2.resize(frame, (600, 400))
  if not success:
    print("Ignoring empty camera frame.")
    continue
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = pose.process(image)
  if results.pose_landmarks:
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, conn, spec)
  cv2.imshow('MediaPipe Pose Youtube', frame)
  if cv2.waitKey(5) & 0xFF == 27:
    break
cap.release()