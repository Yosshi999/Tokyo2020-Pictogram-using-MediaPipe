#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
from dataclasses import dataclass
import json

import cv2 as cv
from mediapipe.python.solutions.pose import PoseLandmark
import numpy as np
import mediapipe as mp
from scipy import stats

from utils import CvFpsCalc, draw_landmarks, draw_stick_figure, minL2

class IdleState:
    def update(self):
        return self
@dataclass
class FocusState:
    rotate_rad: float
    center: tuple
    radius_begin: float
    radius: float
    max_tick: int = 10
    tick: int = 0
    def update(self):
        self.tick += 1
        if self.tick >= self.max_tick:
            return StopState(self.rotate_rad, self.center, self.radius)
        else:
            return self
    @staticmethod
    def try_focus(image, landmarks, current_state, matching):
        landmark_point = []
        for landmark in landmarks:
            if landmark.visibility <= 0.5:
                continue
            landmark_x = min(int(landmark.x * image.shape[1]), image.shape[1] - 1)
            landmark_y = min(int(landmark.y * image.shape[0]), image.shape[0] - 1)
            landmark_point.append([landmark_x, landmark_y])
        if len(landmark_point) > 1:
            # (center_x, center_y), radius = cv.minEnclosingCircle(points=np.array(landmark_point))
            center_x, center_y = matching.movingCenterX * image.shape[1], matching.movingCenterY * image.shape[0]
            radius = np.max(np.linalg.norm(np.array(landmark_point) - np.array([center_x, center_y]), axis=1))

            radius_begin = np.max(np.linalg.norm(
                np.array([
                    [0, 0], [0, image.shape[1]], [image.shape[0], 0], image.shape[:2]
                ])
                - np.array([center_x, center_y])
            , axis=1))
            state = FocusState(-matching.angle, (int(center_x), int(center_y)), radius_begin, radius*1.1)
            return state
        else:
            return current_state
@dataclass
class StopState:
    rotate_rad: float
    center: tuple
    radius: float
    max_tick: int = 30
    tick: int = 0
    def update(self):
        self.tick += 1
        if self.tick >= self.max_tick:
            return IdleState()
        else:
            return self

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=360)

    parser.add_argument('--static_image_mode', action='store_true')
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--rev_color', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    static_image_mode = args.static_image_mode
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    rev_color = args.rev_color

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # pose matching ###########################################################
    with open("graph.json") as f:
        graphs = json.load(f)
    PoseLandmark = mp_pose.PoseLandmark
    target_indices = [
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.LEFT_ELBOW,
        PoseLandmark.RIGHT_ELBOW,
        PoseLandmark.LEFT_WRIST,
        PoseLandmark.RIGHT_WRIST,
        # PoseLandmark.LEFT_HIP,
        # PoseLandmark.RIGHT_HIP,
        # PoseLandmark.LEFT_KNEE,
        # PoseLandmark.RIGHT_KNEE,
        # PoseLandmark.LEFT_ANKLE,
        # PoseLandmark.RIGHT_ANKLE
    ]
    pose_names = []
    data = []
    for graph in graphs:
        name = list(graph.keys())[0]
        positions_raw = graph[name]
        positions = {}
        for p in positions_raw:
            joint_name = list(p.keys())[0]
            positions[joint_name] = p[joint_name]
        data.append(np.array([
            positions["left shoulder"][:2],
            positions["right shoulder"][:2],
            positions["left elbow"][:2],
            positions["right elbow"][:2],
            positions["left hand"][:2],
            positions["right hand"][:2],
        ]))
        pose_names.append(name)
    print(data)

    # ステート管理 #############################################################
    state = IdleState()

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 色指定
    if rev_color:
        color = (255, 255, 255)
        bg_color = (100, 33, 3)
    else:
        color = (100, 33, 3)
        bg_color = (255, 255, 255)

    confidence = np.zeros(len(pose_names))
    prev_pose_name = ""
    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image_fetch = cap.read()
        if not ret:
            break

        if isinstance(state, IdleState):
            image = copy.deepcopy(image_fetch)
            # image = cv.flip(image_fetch, 1)  # ミラー表示したものに更新
        debug_image01 = copy.deepcopy(image)
        debug_image02 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        cv.rectangle(debug_image02, (0, 0), (image.shape[1], image.shape[0]),
                    bg_color,
                    thickness=-1)

        # 検出実施 #############################################################
        if isinstance(state, IdleState):
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            sigma = 200.0
            next_confidence = np.zeros_like(confidence)
            if results.pose_landmarks is not None:
                landmarks = results.pose_landmarks.landmark
                landmarks_pos = np.array([[landmarks[i].x, landmarks[i].y] for i in target_indices])
                visibilities = np.array([landmarks[i].visibility for i in target_indices])
                if visibilities.min() > 0.5:
                    for pose_index, target in enumerate(data):
                        # landmarks X から target Y へのSim(2)変換を推定する
                        opt = minL2(landmarks_pos, target)
                        if opt.scale > 0 and opt.cost > 0:
                            next_confidence[pose_index] = stats.norm.sf(opt.cost, 0, sigma) * 2
                            # confidence[pose_index] = opt.cost
            confidence = confidence * 0.8 + next_confidence * 0.2
            ranking = np.argsort(confidence)[::-1]

        # 描画 ################################################################
        if results.pose_landmarks is not None:
            # 描画
            debug_image01 = draw_landmarks(
                debug_image01,
                results.pose_landmarks,
            )
            debug_image02 = draw_stick_figure(
                debug_image02,
                results.pose_landmarks,
                color=color,
                bg_color=bg_color,
            )

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # if key == ord('f') and isinstance(state, IdleState) and results.pose_landmarks is not None:
        #     state = FocusState.try_focus(image, results.pose_landmarks.landmark, state, None)
        if (confidence[ranking[0]] > 0.3 and prev_pose_name != pose_names[ranking[0]]
                and isinstance(state, IdleState) and results.pose_landmarks is not None):
            print(opt)
            state = FocusState.try_focus(image, results.pose_landmarks.landmark, state, matching=opt)
            if isinstance(state, FocusState):
                prev_pose_name = pose_names[ranking[0]]

        # ステート管理 #########################################################
        state = state.update()
        if isinstance(state, FocusState):
            M = cv.getRotationMatrix2D(state.center, state.rotate_rad * 180 / np.pi * state.tick / state.max_tick, 1)
            debug_image02 = cv.warpAffine(debug_image02, M, (image.shape[1], image.shape[0]), borderValue=(255,255,255))
            radius = int(state.radius_begin + (state.radius - state.radius_begin) * state.tick / state.max_tick)
            mask = np.zeros(debug_image02.shape[:2], dtype=np.uint8)
            cv.circle(mask, state.center, radius, 255, -1)
            debug_image02[mask == 0] = 0
        elif isinstance(state, StopState):
            M = cv.getRotationMatrix2D(state.center, state.rotate_rad * 180 / np.pi, 1)
            debug_image02 = cv.warpAffine(debug_image02, M, (image.shape[1], image.shape[0]), borderValue=(255,255,255))
            radius = int(state.radius)
            mask = np.zeros(debug_image02.shape[:2], dtype=np.uint8)
            cv.circle(mask, state.center, radius, 255, -1)
            debug_image02[mask == 0] = 0
        
        # if isinstance(state, FocusState) or isinstance(state, StopState):
        #     cv.putText(debug_image02, prev_pose_name, (state.center[0], state.center[1]),
        #            cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)

        # 画面反映 #############################################################
        debug_image01 = cv.flip(debug_image01, 1)  # ミラー表示したものに更新
        cv.putText(debug_image01, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(debug_image02, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2, cv.LINE_AA)
        
        for i in range(len(pose_names)):
            r = ranking[i]
            cv.rectangle(debug_image02, (10, 50+20*i-10), (10+400, 50+20*i+10), (0, 255, 0), 1)
            cv.rectangle(debug_image02, (10, 50+20*i-10), (int(10+400*confidence[r]), 50+20*i+10), (0, 255, 0), -1)
            cv.putText(debug_image02, f"{pose_names[r]}: {confidence[r]:.3e}", (10, 50+20*i),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

        cv.imshow('Tokyo2020 Debug', debug_image01)
        cv.imshow('Tokyo2020 Pictogram', debug_image02)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
