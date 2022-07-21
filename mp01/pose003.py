import cv2
import mediapipe as mp
import numpy as np
import math

#pose = mp.solutions.pose.Pose()
#ENABLE_SEGMENTATION: 去背 ;
pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True, smooth_segmentation=True,
                              enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
#連結 Pose 間的點, 若要畫半身, 可以自己建 半身的 list

conn = mp.solutions.pose.POSE_CONNECTIONS
print("Conn:{}".format(conn))

#把 pose 中的點和線畫出來
mp_drawing = mp.solutions.drawing_utils

#pose 中的點和線 的顏色和大小
#使用官方的點.線 style
#spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

#設定個人化的線
spec = mp_drawing.DrawingSpec(color=(255, 255, 255),thickness=3, circle_radius=1)

switch, count = 0, 0
color = (0,0,255)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    h, w, c = frame.shape

    ## xx1 計算 bar 起始位置 : 0.1
    xx1 = int(w * 0.1)
    poslist = []

    #如果有偵測到 pose 時 , results.pose_landmarks 共 0-33 , 共 34 個點
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, conn, spec)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            poslist.append([id, cx, cy])
    try:
        # 右手肘的角度 ; 12: 右肩  14:右手臂; 16:右上臂
        x1, y1 = poslist[12][1], poslist[12][2]
        x2, y2 = poslist[14][1], poslist[14][2]
        x3, y3 = poslist[16][1], poslist[16][2]

        right_angle = abs(int(math.degrees(math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2))))

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.line(frame, (x3, y3), (x2, y2), (0, 255, 255), 3)
        cv2.circle(frame, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, (x1, y1), 15, (0, 255, 255), 2)
        cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 15, (0, 0, 255), 2)
        cv2.circle(frame, (x3, y3), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, (x3, y3), 15, (0, 255, 255), 2)

        # 以10到170度 來計算右手彎曲的程度，最高%=100，最低%=0
        right_per = np.interp(right_angle, (10, 170), (100, 0))

        # 根據右手彎曲程度計算bar的高度 Y軸座標，最高y=200，最低y=400
        right_bar = int(np.interp(right_angle, (10, 170), (200, 400)))

        # 畫矩形來代表bar的高度， 同時印出數字
        cv2.rectangle(frame, (xx1, int(right_bar)), (xx1 + 30, 400), color, cv2.FILLED)
        cv2.putText(frame, str(int(right_per)) + '%', (xx1 - 10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 手起到95%或5%算半個
        color = (0, 0, 255)
        if right_per >= 95:
            color = (0, 255, 0)
            if switch == 0:
                count += 0.5
                switch = 1
        if right_per <= 5:
            color = (0, 255, 0)
            if switch == 1:
                count += 0.5
                switch = 0
    except:
        pass
    cv2.putText(frame, str(count), (xx1 - 40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
    cv2.imshow('MediaPipe Pose Workout', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
