# -*- coding: utf-8 -*-
"""
工具函数模块
包含角度计算、关键点绘制、预处理等核心工具
"""
import numpy as np
import cv2
import mediapipe as mp
from config import *


def calculate_angle(a, b, c):
    """
    计算三个点构成的角度（度数）
    b 为顶点
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def extract_selected_landmarks(landmarks, selected_indices=SELECTED_LANDMARKS):
    """
    从 MediaPipe 33 个关键点中提取选定的 17 个关键点
    返回: (N, 3) 坐标数组, (N,) 可见度数组
    """
    coords = []
    visibilities = []
    for idx in selected_indices:
        lm = landmarks[idx]
        coords.append([lm.x, lm.y, lm.z])
        vis_val = lm.visibility if hasattr(lm, "visibility") and lm.visibility is not None else (
            lm.presence if hasattr(lm, "presence") and lm.presence is not None else 1.0
        )
        visibilities.append(vis_val)
    return np.array(coords, dtype=np.float32), np.array(visibilities, dtype=np.float32)


def smooth_keypoints(current, previous, alpha=SMOOTH_ALPHA):
    """
    指数滑动平均平滑关键点
    p_t_smooth = alpha * p_t + (1 - alpha) * p_{t-1}_smooth
    """
    if previous is None:
        return current.copy()
    return alpha * current + (1 - alpha) * previous


def normalize_keypoints(coords):
    """
    相对坐标归一化：以双髋中心为原点，双肩 2D 距离为尺度
    coords: (17, 3)

    注意：尺度只用 x/y 两维计算。MediaPipe 的 z 是相对深度估算，
    精度远低于 x/y 且跨帧噪声大；把 z 纳入 shoulder_dist 会在人物
    侧身时引入虚假的尺度扰动，污染所有归一化后的坐标。
    """
    # 双髋中心
    hip_center = (coords[KP_LEFT_HIP] + coords[KP_RIGHT_HIP]) / 2.0
    # 双肩 2D 距离（只用 x/y，避开 z 的噪声）
    shoulder_dist = np.linalg.norm(
        coords[KP_LEFT_SHOULDER, :2] - coords[KP_RIGHT_SHOULDER, :2]
    )
    if shoulder_dist < 1e-6:
        shoulder_dist = 1e-6

    # 平移和缩放
    normalized = (coords - hip_center) / shoulder_dist
    return normalized


# 核心关节：首帧若任一不可靠则整帧丢弃
# （鼻、脚跟、脚尖为"装饰性"关键点，不参与任何角度/归一化，可以容忍缺失）
_CRITICAL_JOINTS = (
    KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
    KP_LEFT_HIP, KP_RIGHT_HIP,
    KP_LEFT_ELBOW, KP_RIGHT_ELBOW,
    KP_LEFT_WRIST, KP_RIGHT_WRIST,
    KP_LEFT_KNEE, KP_RIGHT_KNEE,
    KP_LEFT_ANKLE, KP_RIGHT_ANKLE,
)


def interpolate_missing(coords, visibilities, prev_coords=None, next_coords=None,
                        threshold=VISIBILITY_THRESHOLD):
    """
    处理可见度低的关键点：插值或沿用上一帧
    改进：
    1. 增加坐标跳变检测，即使可见度达标但坐标跳变过大也沿用上一帧
    2. 首帧保护：无 prev_coords 可沿用时，若核心关节不可靠则整帧丢弃（返回 None）
       —— 避免把无法修复的坏坐标污染进时序数据

    返回：处理后的 coords (N, 3)，或 None（首帧核心关节不可靠时）
    """
    # 首帧（无 prev/next fallback）的核心关节健壮性检查
    # 当核心关节任一可见度不达标，返回 None 让上游跳过这一帧
    if prev_coords is None and next_coords is None:
        for idx in _CRITICAL_JOINTS:
            if visibilities[idx] < threshold:
                return None

    result = coords.copy()
    # 坐标跳变阈值（归一化坐标下，0.15约为肩宽的一半）
    COORD_JUMP_THRESHOLD = 0.15
    for i in range(len(visibilities)):
        use_prev = False
        if visibilities[i] < threshold:
            use_prev = True
        elif prev_coords is not None:
            # 即使可见度达标，如果坐标跳变过大也视为异常
            dist = np.linalg.norm(coords[i, :2] - prev_coords[i, :2])
            if dist > COORD_JUMP_THRESHOLD:
                use_prev = True

        if use_prev:
            if prev_coords is not None and next_coords is not None:
                # 线性插值
                result[i] = (prev_coords[i] + next_coords[i]) / 2.0
            elif prev_coords is not None:
                # 沿用上一帧
                result[i] = prev_coords[i]
            # 首帧非核心关节（鼻/脚跟/脚尖）若不可靠，保持原值不管
    return result


def should_reuse_previous_pose(coords, visibilities, prev_coords=None,
                               threshold=VISIBILITY_THRESHOLD):
    """
    Frame-level pose sanity check.
    If torso visibility/body span collapses or the whole pose jumps too far,
    keep the previous stable pose instead of accepting a bad frame.
    """
    if prev_coords is None:
        return False

    torso_indices = (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP)
    torso_visible = sum(visibilities[idx] >= threshold for idx in torso_indices)

    shoulder_span = np.linalg.norm(coords[KP_LEFT_SHOULDER, :2] - coords[KP_RIGHT_SHOULDER, :2])
    hip_span = np.linalg.norm(coords[KP_LEFT_HIP, :2] - coords[KP_RIGHT_HIP, :2])
    body_span = max(float(shoulder_span), float(hip_span))

    if torso_visible < 3 or body_span < 0.05:
        return True

    hip_center = (coords[KP_LEFT_HIP, :2] + coords[KP_RIGHT_HIP, :2]) / 2.0
    prev_hip_center = (prev_coords[KP_LEFT_HIP, :2] + prev_coords[KP_RIGHT_HIP, :2]) / 2.0
    prev_shoulder_span = np.linalg.norm(
        prev_coords[KP_LEFT_SHOULDER, :2] - prev_coords[KP_RIGHT_SHOULDER, :2]
    )
    prev_hip_span = np.linalg.norm(
        prev_coords[KP_LEFT_HIP, :2] - prev_coords[KP_RIGHT_HIP, :2]
    )
    prev_body_span = max(float(prev_shoulder_span), float(prev_hip_span), 1e-3)
    max_center_jump = max(0.12, prev_body_span * 1.25)

    return np.linalg.norm(hip_center - prev_hip_center) > max_center_jump


def compute_angle_features(coords):
    """
    计算辅助角度特征
    coords: (17, 3) 归一化后的关键点坐标
    返回: 6 维角度特征向量
    """
    # 左肘角: 左肩-左肘-左腕
    left_elbow_angle = calculate_angle(
        coords[KP_LEFT_SHOULDER], coords[KP_LEFT_ELBOW], coords[KP_LEFT_WRIST]
    )
    # 右肘角: 右肩-右肘-右腕
    right_elbow_angle = calculate_angle(
        coords[KP_RIGHT_SHOULDER], coords[KP_RIGHT_ELBOW], coords[KP_RIGHT_WRIST]
    )
    # 左膝角: 左髋-左膝-左踝
    left_knee_angle = calculate_angle(
        coords[KP_LEFT_HIP], coords[KP_LEFT_KNEE], coords[KP_LEFT_ANKLE]
    )
    # 右膝角: 右髋-右膝-右踝
    right_knee_angle = calculate_angle(
        coords[KP_RIGHT_HIP], coords[KP_RIGHT_KNEE], coords[KP_RIGHT_ANKLE]
    )
    # 髋部夹角: 左膝-左髋-右髋-右膝 (用左膝-髋中心-右膝近似)
    hip_center = (coords[KP_LEFT_HIP] + coords[KP_RIGHT_HIP]) / 2.0
    hip_angle = calculate_angle(
        coords[KP_LEFT_KNEE], hip_center, coords[KP_RIGHT_KNEE]
    )
    # 躯干倾角: 肩中心-髋中心与垂直方向的夹角
    shoulder_center = (coords[KP_LEFT_SHOULDER] + coords[KP_RIGHT_SHOULDER]) / 2.0
    hip_center = (coords[KP_LEFT_HIP] + coords[KP_RIGHT_HIP]) / 2.0
    vertical = np.array([0, -1, 0])  # y轴向上（mediapipe坐标系y向下，取反）
    torso_angle = calculate_angle(
        shoulder_center, hip_center, hip_center + vertical
    )

    # 归一化到 [0, 1] 范围
    angles = np.array([
        left_elbow_angle / 180.0,
        right_elbow_angle / 180.0,
        left_knee_angle / 180.0,
        right_knee_angle / 180.0,
        hip_angle / 180.0,
        torso_angle / 180.0,
    ], dtype=np.float32)

    return angles


def build_frame_feature(coords):
    """
    构建单帧特征向量
    coords: (17, 3) 归一化后的关键点
    返回: 57 维特征向量 (51 坐标 + 6 角度)
    """
    coord_flat = coords.flatten()  # 51维
    angle_feat = compute_angle_features(coords)  # 6维
    return np.concatenate([coord_flat, angle_feat])


# ==================== 字体缓存 ====================
_font_cache = {}


def _get_font(font_size):
    """获取缓存的字体对象，避免每次重新加载"""
    if font_size not in _font_cache:
        try:
            from PIL import ImageFont
            try:
                _font_cache[font_size] = ImageFont.truetype("msyh.ttc", font_size)
            except Exception:
                try:
                    _font_cache[font_size] = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
                except Exception:
                    _font_cache[font_size] = ImageFont.load_default()
        except ImportError:
            _font_cache[font_size] = None
    return _font_cache[font_size]


# ==================== 可视化工具 ====================

# MediaPipe 骨骼连接（基于17点子集的索引）
SKELETON_CONNECTIONS = [
    # 躯干
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, KP_LEFT_HIP),
    (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_RIGHT_HIP),
    # 左臂
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW),
    (KP_LEFT_ELBOW, KP_LEFT_WRIST),
    # 右臂
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    (KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
    # 左腿
    (KP_LEFT_HIP, KP_LEFT_KNEE),
    (KP_LEFT_KNEE, KP_LEFT_ANKLE),
    (KP_LEFT_ANKLE, KP_LEFT_HEEL),
    (KP_LEFT_ANKLE, KP_LEFT_FOOT),
    # 右腿
    (KP_RIGHT_HIP, KP_RIGHT_KNEE),
    (KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
    (KP_RIGHT_ANKLE, KP_RIGHT_HEEL),
    (KP_RIGHT_ANKLE, KP_RIGHT_FOOT),
    # 头部
    (KP_NOSE, KP_LEFT_SHOULDER),
    (KP_NOSE, KP_RIGHT_SHOULDER),
]


def draw_skeleton(frame, coords_pixel, visibilities=None, threshold=0.5):
    """
    在视频帧上绘制骨骼连线和关键点
    coords_pixel: (17, 2) 像素坐标
    """
    h, w = frame.shape[:2]

    # 绘制连接线
    for (i, j) in SKELETON_CONNECTIONS:
        if visibilities is not None:
            if visibilities[i] < threshold or visibilities[j] < threshold:
                continue
        p1 = np.clip(coords_pixel[i], 0.0, 1.0)
        p2 = np.clip(coords_pixel[j], 0.0, 1.0)
        pt1 = (int(p1[0] * w), int(p1[1] * h))
        pt2 = (int(p2[0] * w), int(p2[1] * h))
        cv2.line(frame, pt1, pt2, (0, 255, 128), 2, cv2.LINE_AA)

    # 绘制关键点
    for i in range(len(coords_pixel)):
        if visibilities is not None and visibilities[i] < threshold:
            continue
        p = np.clip(coords_pixel[i], 0.0, 1.0)
        pt = (int(p[0] * w), int(p[1] * h))
        cv2.circle(frame, pt, 5, (0, 128, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_info_panel(frame, action_name, count, score, tips, fps=0):
    """
    在视频帧上绘制信息面板
    优化：所有文字统一一次PIL转换，避免每个文字单独转换
    """
    h, w = frame.shape[:2]

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    try:
        from PIL import Image, ImageDraw
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        font_normal = _get_font(int(0.7 * 30))
        font_small = _get_font(int(0.5 * 30))
        font_tip = _get_font(int(0.6 * 30))

        # 注意：PIL用RGB颜色，OpenCV传入的是BGR，需要反转
        draw.text((10, 30), f"动作: {action_name}", font=font_normal, fill=(128, 255, 0))
        draw.text((250, 30), f"次数: {count}", font=font_normal, fill=(0, 255, 255))
        draw.text((400, 30), f"评分: {score:.0f}", font=font_normal, fill=(255, 200, 0))
        draw.text((550, 30), f"FPS: {fps:.1f}", font=font_small, fill=(200, 200, 200))
        if tips:
            draw.text((10, 65), f"提示: {tips}", font=font_tip, fill=(0, 0, 255))

        result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        np.copyto(frame, result)
    except ImportError:
        cv2.putText(frame, f"Action: {action_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Count: {count}", (250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Score: {score:.0f}", (400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (550, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2, cv2.LINE_AA)
        if tips:
            cv2.putText(frame, f"Tips: {tips}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return frame


def put_chinese_text(img, text, position, color=(255, 255, 255), font_scale=0.7):
    """
    在图像上绘制文本（使用OpenCV，中文可能需要PIL支持）
    优先尝试PIL，失败则回退到cv2
    注意：如果需要在同一帧上绘制多段文字，请使用 draw_info_panel 以避免多次转换
    """
    font_size = int(font_scale * 30)
    font = _get_font(font_size)

    if font is not None:
        try:
            from PIL import Image, ImageDraw
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text(position, text, font=font, fill=color[::-1])
            result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            np.copyto(img, result)
            return img
        except ImportError:
            pass

    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 2, cv2.LINE_AA)
    return img
