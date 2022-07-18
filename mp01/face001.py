import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
# 載入嘴唇的透明背景圖片
mouth_normal = cv2.imread("m6.png")
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)

#while(True):
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    # 先找出畫面的長寬大小
    h, w, d = frame.shape
    print('image height:{} , width={}'.format(h,w))

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            try:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())

                # 點0與17分別是嘴唇上下的座標，取得嘴唇大小 ; 座標點. 是用 比例來算的. 因為 windows 視窗大小會不同. 所以用比例算
                mouth_len = int((face_landmarks.landmark[17].y * h)-(face_landmarks.landmark[0].y * h))
                print(mouth_len)

                # 將嘴唇圖案的圖片轉換成適合的大小
                mouth = cv2.resize(mouth_normal, (mouth_len * 3, mouth_len), cv2.INTER_AREA)
                img_height, img_width, _ = mouth.shape
                print(mouth.shape)
                x, y = int(face_landmarks.landmark[13].x * w - img_width/2), \
                       int(((face_landmarks.landmark[13].y+face_landmarks.landmark[14].y)/2) * h - img_height/2)
                frame[y: y + img_height, x: x + img_width] = mouth

            except:
                pass
    cv2.imshow("face_mesh2", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
