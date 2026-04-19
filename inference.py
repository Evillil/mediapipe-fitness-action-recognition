# -*- coding: utf-8 -*-
"""
实时推理引擎
滑动窗口实时动作识别
适配 MediaPipe 0.10+ Tasks API
"""
import os
import numpy as np
import torch
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from model import ActionRecognitionModel
from config import *
from utils import (extract_selected_landmarks, smooth_keypoints, normalize_keypoints,
                   interpolate_missing, should_reuse_previous_pose, build_frame_feature)


class ActionInference:
    """
    动作实时推理引擎
    维护30帧关键点缓存队列，满足窗口要求后进行推理
    """

    def __init__(self, model_path=None, device=None, predict_enabled=True):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "best_cnn_lstm.pth")

        self.model = None
        self.feature_dim = 57
        self.predict_enabled = predict_enabled
        if self.predict_enabled:
            self._load_model(model_path)

        # 关键点缓存 — 使用固定大小 numpy 环形缓冲区，避免每帧 deque→array 拷贝
        self.keypoint_buffer = np.zeros((WINDOW_SIZE, self.feature_dim), dtype=np.float32)
        self.buffer_idx = 0
        self.buffer_count = 0
        self.prev_smoothed = None

        # 预测结果平滑
        self.prediction_history = deque(maxlen=PREDICTION_SMOOTH_WINDOW)

        # 上一帧坐标（使用平滑后的坐标用于插值，减少跳变）
        self.last_raw_coords = None
        self.last_smoothed_coords = None

    def _load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            print(f"[警告] 模型文件未找到: {model_path}")
            return

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        if 'conv1.0.weight' in state_dict:
            self.feature_dim = state_dict['conv1.0.weight'].shape[1]

        self.model = ActionRecognitionModel(input_dim=self.feature_dim).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"模型加载成功: {model_path} (特征维度: {self.feature_dim})")

    def reset(self):
        """重置推理状态"""
        self.keypoint_buffer[:] = 0
        self.buffer_idx = 0
        self.buffer_count = 0
        self.prev_smoothed = None
        self.prediction_history.clear()
        self.last_raw_coords = None
        self.last_smoothed_coords = None

    def process_landmarks_from_result(self, pose_result):
        """
        处理 PoseLandmarker 结果（新API）
        pose_result: PoseLandmarkerResult
        返回: (action_class, confidence, raw_coords_17x2, visibilities_17, raw_3d_coords_17x3)
        """
        if not pose_result.pose_landmarks or len(pose_result.pose_landmarks) == 0:
            return None, 0.0, None, None, None

        landmarks = pose_result.pose_landmarks[0]

        coords, vis = extract_selected_landmarks(landmarks)

        # 可见度过滤与插值（使用平滑后的历史坐标作为回退基准，减少跳变）
        fallback_coords = self.last_smoothed_coords if self.last_smoothed_coords is not None else self.last_raw_coords
        if should_reuse_previous_pose(coords, vis, prev_coords=fallback_coords):
            coords = fallback_coords.copy()
            vis = np.maximum(vis, VISIBILITY_THRESHOLD)
        coords = interpolate_missing(coords, vis, prev_coords=fallback_coords)
        # 首帧核心关节不可靠时 interpolate_missing 会返回 None，整帧丢弃
        # 等待下一帧再建立 fallback 基准
        if coords is None:
            return None, 0.0, None, None, None
        self.last_raw_coords = coords.copy()

        # 指数滑动平均平滑
        smoothed = smooth_keypoints(coords, self.prev_smoothed)
        self.prev_smoothed = smoothed.copy()
        self.last_smoothed_coords = smoothed.copy()
        stable_coords = smoothed.copy()

        # 固定动作模式下只需要关键点，不再执行动作分类预测
        raw_pixel = stable_coords[:, :2]
        if not self.predict_enabled:
            return None, 0.0, raw_pixel, vis, stable_coords

        # 归一化
        normed = normalize_keypoints(smoothed)

        # 构建特征
        feature = build_frame_feature(normed)

        # 添加到环形缓冲区
        self.keypoint_buffer[self.buffer_idx] = feature
        self.buffer_idx = (self.buffer_idx + 1) % WINDOW_SIZE
        self.buffer_count = min(self.buffer_count + 1, WINDOW_SIZE)

        # 像素坐标（用于绘制）
        raw_pixel = stable_coords[:, :2]

        # 缓存不足，无法推理
        if self.buffer_count < WINDOW_SIZE:
            return None, 0.0, raw_pixel, vis, stable_coords

        # 执行推理
        action_class, confidence = self._predict()

        return action_class, confidence, raw_pixel, vis, stable_coords

    def _predict(self):
        """执行模型推理"""
        if self.model is None:
            return None, 0.0

        # 从环形缓冲区按正确时间顺序取出窗口
        window = np.roll(self.keypoint_buffer, -self.buffer_idx, axis=0)
        input_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        pred_class = predicted.item()
        conf = confidence.item()

        # 结果平滑（多数投票）
        self.prediction_history.append(pred_class)

        if len(self.prediction_history) >= 3:
            vote = Counter(self.prediction_history)
            smoothed_class = vote.most_common(1)[0][0]
        else:
            smoothed_class = pred_class

        return smoothed_class, conf

    def get_action_name(self, class_idx):
        """获取动作中文名称"""
        if class_idx is None:
            return "等待中..."
        action_key = IDX_TO_CLASS.get(class_idx, "unknown")
        return ACTION_NAMES_CN.get(action_key, "未知")

    def get_action_key(self, class_idx):
        """获取动作英文键名"""
        if class_idx is None:
            return None
        return IDX_TO_CLASS.get(class_idx, None)
