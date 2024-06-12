# 웹캠으로 피아노를 치는 손을 추적하고 해당 정보를 유니티로 전송하여 VR 씬에서 손 모델을 움직이는데 사용하기 위한 코드

import cv2
import mediapipe as mp
import socket
import cv2.aruco as aruco
import numpy as np


# 손 위치 추적을 위한 함수
def hand_tracking(img, hand):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hand.process(img_RGB)

    hand_pos_data = []

    # 화면에 손이 있다면 실행
    if results.multi_hand_landmarks:
        cur_hand_x = 0
        prev_hand_x = 0

        for hand_landmarks in results.multi_hand_landmarks:
            # mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cur_hand_pos_data = []

            for id, landmark in enumerate(hand_landmarks.landmark):
                # 손 랜드마크의 위치를 화면 좌표로 변환
                h, w, c = img.shape
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                
                # 손목 좌표를 손의 대표 좌표로 지정
                if id == 0:
                    cur_hand_x = cx
                # 각 손가락 끝의 좌표와 손목 좌표, 중지의 시작 좌표를 저장 (중지의 시작 좌표는 손의 방향 계산에 사용)
                if id == 0 or id == 9 or id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                    cur_hand_pos_data.append((landmark.x * w, landmark.y * h))

            # 오른손 좌표가 먼저 오게 만들어 줌
            if cur_hand_x > prev_hand_x:
                hand_pos_data = cur_hand_pos_data + hand_pos_data
            else:
                hand_pos_data += cur_hand_pos_data
            prev_hand_x = cur_hand_x

    return hand_pos_data


# 마커 추적을 위한 함수
def marker_tracking(img, aruco_dict):
    # 마커를 추적하여 꼭지점 좌표와 인덱스 반환
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bboxs, ids, rejected = aruco.detectMarkers(img_gray, aruco_dict)
    return bboxs, ids


# 두 점 사이의 거리를 계산하는 함수
def get_distance(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]

    return (x ** 2 + y ** 2) ** 0.5


# 두 벡터 사이의 각을 계산하는 함수
def get_angle(vector1, vector2):
    # 두 벡터의 내적을 계산
    dot_product = np.dot(vector1, vector2)

    # 두 벡터의 크기를 계산
    norm_v1 = np.linalg.norm(vector1)
    norm_v2 = np.linalg.norm(vector2)

    # 코사인 법칙을 사용하여 두 벡터 사이의 각을 계산
    cos_theta = dot_product / (norm_v1 * norm_v2)

    # acos 함수를 사용하여 각도를 계산 (라디안 단위)
    angle_rad = np.arccos(cos_theta)

    return angle_rad


# 마커와 손 랜드마크 좌표를 인풋으로 받아 마커 좌표를 기준으로 손 랜드마크 좌표를 만들어 주는 함수
# 유니티에서 이 좌표를 바탕으로 VR 씬에서 손의 위치를 조정
def get_processed_data(start_point, end_point, edge_point, pos_data):
    output = ''
    
    # 마커 영역의 가로 세로 길이를 계산
    x_length = get_distance(start_point, edge_point)
    y_length = get_distance(edge_point, end_point)

    # 마커를 기준으로 손 랜드마크 좌표 계산
    vector1 = np.array([edge_point[0] - start_point[0], -edge_point[1] + start_point[1]])
    
    for pos in pos_data:

        vector2 = np.array([pos[0] - start_point[0], -pos[1] + start_point[1]])

        p_length = get_distance(start_point, pos)
        angle = get_angle(vector1, vector2)

        x = round(p_length * np.cos(angle) / x_length, 4)

        y = round(p_length * np.sin(angle) / y_length, 4)

        if pos[1] > start_point[1]:
            y *= -1

        output += str(x)
        output += ' '
        output += str(y)
        output += ' '

    return output


if __name__ == "__main__":
    # 유니티로 손 좌표 정보를 보내기 위한 UDP 소캣 생성
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverAddressPort = ("127.0.0.1", 5052)

    # 손 추적을 위한 mediapipe 모듈
    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hand = mp_hands.Hands()
    
    # 마커 추적을 위한 딕셔너리 
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # 웹캡 화면에서 손과 마커를 추적하고 마커를 기준으로한 손 랜드마커 좌표를 계산하여 유니티로 전송
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    while True:
        success, img = cap.read()
        
        # 마커 추적
        bboxs, ids = marker_tracking(img, aruco_dict)
        if len(bboxs) > 0:
            for id, bbox in enumerate(bboxs):
                if ids[id][0] == 2:
                    sp = (int(bbox[0][1][0]), int(bbox[0][1][1]))
                elif ids[id][0] == 3:
                    ep = (int(bbox[0][0][0]), int(bbox[0][0][1]))
                elif ids[id][0] == 0:
                    edge = (int(bbox[0][0][0]), int(bbox[0][0][1]))
        
        # 손 추적
        hand_pos_data = hand_tracking(img, hand)
        
        # 마커를 기준으로 손 랜드마커 좌표 계산 및 전송
        if sp is not None and ep is not None and edge is not None:
            processed_data = get_processed_data(sp, ep, edge, hand_pos_data)
            sock.sendto(str.encode(str(processed_data)), serverAddressPort)

        # cv2.imshow("Image", img)
        cv2.waitKey(1)

