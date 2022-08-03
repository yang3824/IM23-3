import mediapipe as mp
import pyautogui as pag #抓取滑鼠座標的工具
import numpy as np
import function
import time
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

function.initialize()

#time = 0

cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():  # 連接鏡頭成功判斷
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR 2 RGB #轉換顏色設定順序：mediapipe讀法為bgr python為rgb
        image = cv2.flip(image, 1) # Flip on horizontal
        image.flags.writeable = False # Set flag

        results = hands.process(image)  # Detections
        image.flags.writeable = True # Set flag to true
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGB 2 BGR

        # print(results) # Detections

        # Rendering results 若有抓到手部座標
        if results.multi_hand_landmarks:
           for num, hand in enumerate(results.multi_hand_landmarks):
               # num為手的數量(0為一隻手，1為兩隻手)，hand為手的座標
               mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(255, 255, 25), thickness=2, circle_radius=2),
                                      )

               # Render left or right detection
               function.get_label(image, num, hand, results)
                   #text, coord = function.get_label(num, hand, results)
                   #cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

               function.center_of_palm(image, num, results)

           # Draw angles to image from joint list
           function.draw_fingertip_coordinate(image, results)
           function.draw_finger_angle(image, results)
           pag.moveTo(function.get("screenWidth") * function.get("lx"), function.get("screenHeight") * function.get("ly"))
           function.action_response()

           #if True: function.draw_hand_model(results)

           if (int(time.time()) - function.get("time")) > 3:
               function.draw_hand_model(results)
               function.set("time", int(time.time()))

           # Save our image
           # cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
           cv2.imshow('Hand Tracking', image)

           if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()