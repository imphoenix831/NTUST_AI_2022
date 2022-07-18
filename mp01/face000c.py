import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True
                                  , min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
while(True):
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,   #mp_face_mesh.FACEMESH_CONTOURS
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()