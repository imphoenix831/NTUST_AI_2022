import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 3, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
<<<<<<< HEAD
=======
    frame = cv2.resize(frame, (1024, 680))

>>>>>>> parent of f1b4f16 (Revert "mediapipe-hand")
    if not success:
        print("Ignoring empty camera frame.")
        continue
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            ,mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow('hand001', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()