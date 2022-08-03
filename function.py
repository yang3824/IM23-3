import mediapipe as mp
import pyautogui as pag  # 抓取滑鼠座標的工具
import numpy as np
import time
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def now_time():
    local_time = time.localtime()  # 取得時間元組
    timeString = time.strftime("%H%M%S", local_time)  # 轉成想要的字串形式
    return timeString

def initialize():
    global coor
    screenWidth, screenHeight = pag.size()
    coor = {"screenWidth": screenWidth, "screenHeight": screenHeight, "lx": 100, "ly": 100, "rx": 100, "ry": 100,
            "action": "", "lasttime": now_time(), "time": int(time.time()), "hand": np.array([]), "l_finger_proportion": np.array([]), "r_finger_proportion": np.array([])}


def set(coo, val):
    coor[coo] = val


def get(want):
    if want not in coor.keys():
        return "error"
    else:
        return coor[want]


def get_label(image, index, hand, results):
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))

            # Extract Coordinates
            coords = tuple(np.multiply(np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x,
                                                 hand.landmark[mp_hands.HandLandmark.WRIST].y)), [1250, 750]).astype(int))

            cv2.putText(image, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


def draw_fingertip_coordinate(image, results):
    # Loop through hands
    joint_list1 = [[4], [8], [12], [16], [20]]
    for hand in results.multi_hand_landmarks:
        # Loop through joint sets
        for joint in joint_list1:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])  # First coord 指尖

            # 顯示座標
            cv2.putText(image, str(round(a[0], 2)) + " ," + str(round(a[1], 2)), tuple(np.multiply(a, [1250, 600]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return image


def draw_finger_angle(image, results):
    # Loop through hands
    joint_list = [[7, 6, 5], [11, 10, 9], [15, 14, 13], [19, 18, 17]]
    for hand in results.multi_hand_landmarks:
        total_angle = 0.0

        # Loop through joint sets
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])  # First coord 指尖
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])  # Second coord 第一關節
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])  # Third coord 第二關節

            # radians（弧度） # arctan2：計算弧度的數學工具
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

            angle = np.abs(radians * 180.0 / np.pi)
            if angle > 180.0: angle = 360 - angle

            total_angle = total_angle + angle

            # 顯示彎曲度
            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(a, [1250, 400]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 127), 2, cv2.LINE_AA)

        if total_angle/4 < 50:
            cv2.putText(image, "make a fist", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            set("action", "fist")
        else:
            set("action", "")
            set("lasttime", now_time())



    return image


def center_of_palm(image, index, results):
    for hand in results.multi_hand_landmarks:
        # xc = float(hand.landmark[9].x + hand.landmark[0].x + hand.landmark[1].x + hand.landmark[7].x) / 4.0
        # yc = float(hand.landmark[9].y + hand.landmark[0].y + hand.landmark[1].y + hand.landmark[7].y) / 4.0
        zc = np.array([hand.landmark[0].x, hand.landmark[0].y])

        xct = float((4 * hand.landmark[0].x + hand.landmark[5].x + hand.landmark[17].x) / 6.0)
        yct = float((4 * hand.landmark[0].y + hand.landmark[5].y + hand.landmark[17].y) / 6.0)

        cv2.putText(image, str(round(xct, 2)) + " ," + str(round(yct, 2)), tuple(np.multiply(zc, [1250, 500]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            # Process results
            label = classification.classification[0].label
            if label == "Left":
                # globals.left_x = round(xc, 2)
                # globals.left_y = round(yc, 2)
                set("lx", round(xct, 2))
                set("ly", round(yct, 2))
                # print("left ", globals.get("lx"), globals.get("ly"))
            elif label == "Right":
                # globals.right_x = round(xc, 2)
                # globals.righr_y = round(yc, 2)
                set("rx", round(xct, 2))
                set("ry", round(yct, 2))
                # print("right ", globals.get("rx"), globals.get("ry"))
            else:
                initialize()


def action_response():
    actions = get("action")
    if actions == "fist":
        if (int(now_time()) - int(get("lasttime"))) < 2:
            pag.click()
        else:
            # pag.dragRel(get("lx"), get("ly"), button='left')
            pag.mouseDown(x=get("screenWidth") * get("lx"), y=get("screenHeight") * get("ly"), button='left')
    elif actions == "":
        pag.mouseUp(x=get("screenWidth") * get("lx"), y=get("screenHeight") * get("ly"), button='left')


def draw_hand_model(results):
    hand_ratio = np.array([])
    joint_list = [[8, 7, 6, 5], [12, 11, 10, 9], [16, 15, 14, 13], [20, 19, 18, 17]]  # 無拇指[4, 3, 2]

    for hand in results.multi_hand_landmarks:
        thumb1 = np.array([10 * hand.landmark[4].x, 10 * hand.landmark[4].y])  # First coord 指尖
        thumb2 = np.array([10 * hand.landmark[3].x, 10 * hand.landmark[3].y])  # Second coord 第一關節
        thumb3 = np.array([10 * hand.landmark[2].x, 10 * hand.landmark[2].y])  # Third coord 第二關節

        thumb1_long = round(np.linalg.norm(thumb1 - thumb2), 2)  # 計算第一指節長度
        thumb2_long = round(np.linalg.norm(thumb2 - thumb3), 2)  # 計算第二指節長度

        hand_ratio = np.array([thumb1_long, thumb2_long, 9.99])

        for hand in results.multi_hand_landmarks:
            # Loop through joint sets
            for joint in joint_list:
                a = np.array([10 * hand.landmark[joint[0]].x, 10 * hand.landmark[joint[0]].y])  # First coord 指尖
                b = np.array([10 * hand.landmark[joint[1]].x, 10 * hand.landmark[joint[1]].y])  # Second coord 第一關節
                c = np.array([10 * hand.landmark[joint[2]].x, 10 * hand.landmark[joint[2]].y])  # Third coord 第二關節
                d = np.array([10 * hand.landmark[joint[3]].x, 10 * hand.landmark[joint[3]].y])  # 掌心上緣關節

                Knuckle1 = round(np.linalg.norm(a - b), 2)  # 計算第一指節長度
                Knuckle2 = round(np.linalg.norm(b - c), 2)  # 計算第二指節長度
                Knuckle3 = round(np.linalg.norm(c - d), 2)  # 計算第三指節長度

                temp1 = np.array([Knuckle1, Knuckle2, Knuckle3])
                hand_ratio = np.vstack([hand_ratio, temp1])

        nine = np.array([10 * hand.landmark[9].x, 10 * hand.landmark[9].y])  # 取得點九座標
        zero = np.array([10 * hand.landmark[0].x, 10 * hand.landmark[0].y])  # 取得點零座標
        palm_height = round(np.sqrt(np.sum(np.square(nine-zero))), 2)  # 計算手賞高度

        five = np.array([10 * hand.landmark[5].x, 10 * hand.landmark[5].y])  # 取得點五座標
        seventeen = np.array([10 * hand.landmark[0].x, 10 * hand.landmark[0].y])  # 取得點十七座標
        palm_width = round(np.sqrt(np.sum(np.square(five - seventeen))), 2)  # 計算手賞寬度

        temp2 = np.array([palm_height, palm_width, 9.99])
        hand_ratio = np.vstack([hand_ratio, temp2])
        set("hand", hand_ratio)

    print("Proportion of fingers:\n", hand_ratio[0:4, :], "\n")
    print("Proportion of Palm:\n", hand_ratio[5, :], "\n")

    help_model(results)

def help_model(results):
    for hand in results.multi_hand_landmarks:
        twelve = np.array([10 * hand.landmark[12].x, 10 * hand.landmark[12].y])
        zero = np.array([10 * hand.landmark[0].x, 10 * hand.landmark[0].y])
        seventeen = np.array([10 * hand.landmark[17].x, 10 * hand.landmark[17].y])
        four = np.array([10 * hand.landmark[4].x, 10 * hand.landmark[4].y])

        height = round(np.sqrt(np.sum(np.square(twelve - zero))), 2)
        width = round(np.sqrt(np.sum(np.square(seventeen - four))), 2)

        print("model:\n", height, width, "\n")