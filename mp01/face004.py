import cv2
import mediapipe as mp
import pafy

# 開啟畫關鍵點與face mesh網格功能
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles

# 載入嘴唇的透明背景圖片
mouth_2 = cv2.imread("m6.png")
mouth_3 = cv2.imread("m7.png")

# 設定正確率
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing_styles = mp.solutions.drawing_styles


###############################################################################
# 設定攝影機
#手語
#url ='https://www.youtube.com/watch?v=hPC7Nw8DcUo'
#url ='https://www.youtube.com/watch?v=wYB9Vu282ZU'

#周震南
url='https://www.youtube.com/watch?v=Rj97LJSSW7M'

live = pafy.new(url)
stream = live.getbest(preftype="mp4")

cap = cv2.VideoCapture(stream.url)
#write video format and size
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('手語.mp4', fourcc, 20, (1024, 680))
'''
# 設定攝影機
cap = cv2.VideoCapture(0)
'''
#while(True):
while cap.isOpened():

    # 從攝影機擷取一禎圖片
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    # 先找出畫面的長寬大小
    frame = cv2.resize(frame, (1024, 680))

    h, w, d = frame.shape
    # Opencv用BGR所以先轉換顏色為RGB並傳到face mesh運算
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        # 超過 1個人時, 改出. m7.png 請務群聚的圖
        if len(results.multi_face_landmarks) >1 :
            mouth_normal = mouth_3
        else:
            mouth_normal = mouth_2

        for face_landmarks in results.multi_face_landmarks:
            try:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    #onnections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())

                #畫出眼睛
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())


                # 點0與17分別是嘴唇上下的座標，取得嘴唇大小
                mouth_len = int((face_landmarks.landmark[17].y * h)-int(face_landmarks.landmark[0].y * h))
                # 將嘴唇圖案的圖片轉換成適合的大小
                mouth = cv2.resize(mouth_normal, (mouth_len * 3, mouth_len))

                # 將嘴唇圖案轉灰階
                mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)

                # 將嘴唇圖案去背
                _, mouth_mask = cv2.threshold(mouth_gray, 25, 255, cv2.THRESH_BINARY_INV)
                # 找出嘴唇的高度img_height 與寬度img_width
                img_height, img_width, _ = mouth.shape

                # 點13與14的中間是嘴唇的中心點，找出放圖的左上角落座標
                x, y = int(face_landmarks.landmark[13].x * w - img_width/2), \
                       int(((face_landmarks.landmark[13].y + face_landmarks.landmark[14].y)/2) * h - img_height/2)

                # 將去背圖案與真的人嘴唇合併成一矩形 mouth
                mouth_area = frame[y: y + img_height, x: x + img_width]
                mouth_area_no_mouth = cv2.bitwise_and(mouth_area, mouth_area, mask=mouth_mask)

                mouth = cv2.add(mouth_area_no_mouth, mouth)
                # 在點(x, y)放上圖案mouth
                frame[y: y+img_height, x: x+img_width] = mouth

            except:
                pass

    cv2.imshow("M10609906 Phoenix", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
