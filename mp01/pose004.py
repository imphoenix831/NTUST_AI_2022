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
switch1, count1 = 0, 0
color = (0,0,255)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #imgRGB  = cv2.flip(frame,1 )  #filp: 0 #上下翻轉,  filp:1 #左右翻轉 filp:-1 #上下左右翻轉
    results = pose.process(imgRGB)
    h, w, c = frame.shape

    ## xx1 計算 bar 起始位置 : 0.1
    xx1 = int(w * 0.1)
    xx2 = int(w * 0.9)
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

        # 左手肘的角度 ; 11: 左肩  13:左手臂; 15:左上臂
        lx1, ly1 = poslist[11][1], poslist[11][2]
        lx2, ly2 = poslist[13][1], poslist[13][2]
        lx3, ly3 = poslist[15][1], poslist[15][2]


        right_angle = abs(int(math.degrees(math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2))))
        left_angle = abs(int(math.degrees(math.atan2(ly1 - ly2, lx1 - lx2) - math.atan2(ly3 - ly2, lx3 - lx2))))

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.line(frame, (x3, y3), (x2, y2), (0, 255, 255), 3)
        cv2.circle(frame, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, (x1, y1), 15, (0, 255, 255), 2)
        cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 15, (0, 0, 255), 2)
        cv2.circle(frame, (x3, y3), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, (x3, y3), 15, (0, 255, 255), 2)

        cv2.line(frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 3)
        cv2.line(frame, (lx3, ly3), (lx2, ly2), (0, 255, 255), 3)
        cv2.circle(frame, (lx1, ly1), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, (lx1, ly1), 15, (0, 255, 255), 2)
        cv2.circle(frame, (lx2, ly2), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(frame, (lx2, ly2), 15, (0, 0, 255), 2)
        cv2.circle(frame, (lx3, ly3), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, (lx3, ly3), 15, (0, 255, 255), 2)

        # 以10到170度 來計算右手彎曲的程度，最高%=100，最低%=0
        right_per = np.interp(right_angle, (10, 170), (100, 0))
        left_per = np.interp(left_angle, (10, 170), (100, 0))

        # 根據右手彎曲程度計算bar的高度 Y軸座標，最高y=200，最低y=400
        right_bar = int(np.interp(right_angle, (10, 170), (200, 400)))
        left_bar = int(np.interp(left_angle, (10, 170), (200, 400)))

        # 畫矩形來代表bar的高度， 同時印出數字
        cv2.rectangle(frame, (xx1, int(right_bar)), (xx1 + 30, 400), color, cv2.FILLED)
        cv2.putText(frame, str(int(right_per)) + '%', (xx1 - 10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


        cv2.rectangle(frame, (xx2, int(left_bar)), (xx2+30 , 400), color, cv2.FILLED)
        cv2.putText(frame, str(int(left_per)) + '%', (xx2-10 , 450), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

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



        color = (0, 0, 255)
        if left_per >= 95:
            color = (0, 255, 0)
            if switch1 == 0:
                count1 += 0.5
                switch1 = 1
        if left_per <= 5:
            color = (0, 255, 0)
            if switch1 == 1:
                count1 += 0.5
                switch1 = 0
    except:
        pass
    cv2.putText(frame, str(count), (xx1 - 40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)

    cv2.putText(frame, str(count1), (xx2-40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)

    cv2.imshow('MediaPipe Pose Workout', frame)

    if (cv2.waitKey(5) & 0xFF == 27) or (cv2.waitKey(5)==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
