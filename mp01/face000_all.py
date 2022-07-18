import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0),thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True      #refine_landmarks=True : 用眼球
                                  , min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 捉攝影機畫面
cap = cv2.VideoCapture(0)
#while(True):
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape

    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,   #網狀連結: FACEMESH_TESSELATION
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,  # mp_face_mesh.FACEMESH_CONTOURS
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,  # mp_face_mesh.FACEMESH_IRISES : 捉眼睛
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())

            '''
            ## 畫出 486 個 face 的點
            for id, lm in enumerate(face_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                # 作業要修改以下三行，想像看如何在每個手指端點畫圓？
                # (0,255,0) -> (藍,綠,紅) , 注意(B,G,R)
                # cv2.circle(frame, (cx, cy), 15, (219,112,147), cv2.FILLED)
                if ((id == 4) or (id == 8) or (id == 12) or (id == 16) or (id == 20)):
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), cv2.FILLED)
                else:
                    cv2.circle(frame, (cx, cy), 2, (0, 0, 0), cv2.FILLED)

                # 語法：cv2.putText(img, text, org, fontFace, fontScale, color[, thickness])
                # 把 數字 cx+8 ,   FONT_HERSHEY_COMPLEX: 3號字體 ; 0.6 字型大小  ;  (0,0,255): 紅色   1:線的粗型
                cv2.putText(frame, str(id), (cx + 8, cy), cv2.FONT_HERSHEY_COMPLEX, .2, (0, 0, 0), 1)
            '''

            cv2.imshow('M10609906', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()