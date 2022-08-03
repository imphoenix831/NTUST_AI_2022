import cv2
import mediapipe as mp
import pafy

#mediapipe 畫圖功能
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode= False, max_num_hands = 6,
                       min_detection_confidence=0.8, min_tracking_confidence=0.8)

mp_drawing_styles = mp.solutions.drawing_styles

#手模
#url = "https://www.youtube.com/watch?v=UdZwXAhDW-s"
#手語
url ='https://www.youtube.com/watch?v=hPC7Nw8DcUo'
#url ='https://www.youtube.com/watch?v=wYB9Vu282ZU'




live = pafy.new(url)
stream = live.getbest(preftype="mp4")

cap = cv2.VideoCapture(stream.url)
#write video format and size
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('手語1.mp4', fourcc, 20, (1024, 680))

#cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()

    #視窗變大些
    frame = cv2.resize(frame, (1024, 680))
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Draw the hand annotations on the image.
    #cv2 三原色: BGR , 不是 RGB
    #要 CV2的 BGR 轉成 其他程式讀 RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape

    #轉成 圖 image 後, 再判斷是否 偵測到 手
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # print(hand_landmarks)
            # 捉出 hand 座標 畫出 手部的連結線 , 並畫在 frame (CV2) 顯示出來
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                                        ,mp_drawing_styles.get_default_hand_landmarks_style(),
                                         mp_drawing_styles.get_default_hand_connections_style())

            for id, lm in enumerate(hand_landmarks.landmark):
                #把 偵測出來的 x, y 和 cv2 圖片大小 , 依比例算出 cx , cy
                cx, cy = int(lm.x * w), int(lm.y * h)
                # 作業要修改以下三行，想像看如何在每個手指端點畫圓？
                    # (0,255,0) -> (藍,綠,紅) , 注意(B,G,R)
                #cv2.circle(frame, (cx, cy), 15, (219,112,147), cv2.FILLED)
                if ((id ==4) or (id ==8) or (id ==12) or (id ==16) or (id ==20)):
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), cv2.FILLED)
                else:
                    cv2.circle(frame, (cx, cy), 2, (0, 0, 0), cv2.FILLED)


                # 語法：cv2.putText(img, text, org, fontFace, fontScale, color[, thickness])
                #把 數字 cx+8 ,   FONT_HERSHEY_COMPLEX: 3號字體 ; 0.6 字型大小  ;  (0,0,255): 紅色   1:線的粗型
                cv2.putText(frame,str(id),(cx+8, cy),cv2.FONT_HERSHEY_COMPLEX,.2,(0,0,0),1)


    cv2.imshow('hand002', frame)

    #將偵測後手的影片儲存起來
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        cv2.imwrite("Hand002.jpg", frame)
        break

#將離開時的影像, 儲存成 output.jpg

out.release()
cap.release()
cv2.destroyAllWindows()