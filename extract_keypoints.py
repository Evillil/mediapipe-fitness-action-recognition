# -*- coding: utf-8 -*-
"""
关键点提取与样本生成模块
从视频中提取MediaPipe关键点，构建滑动窗口时序样本
包含数据增强功能
适配 MediaPipe 0.10+ Tasks API
"""
import os
import sys
import json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from config import *
from utils import (extract_selected_landmarks, smooth_keypoints,
                   normalize_keypoints, interpolate_missing,
                   build_frame_feature, compute_angle_features,
                   should_reuse_previous_pose)


def create_pose_landmarker(model_path=None):
    """创建 PoseLandmarker 实例 (IMAGE模式，无状态，适合批量处理)"""
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "pose_landmarker_heavy.task")
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_landmarks_from_result(result):
    """
    从 PoseLandmarker 结果中提取关键点
    新API返回 result.pose_landmarks (list of list of NormalizedLandmark)
    """
    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None, None

    landmarks = result.pose_landmarks[0]  # 取第一个人

    return extract_selected_landmarks(landmarks)


def extract_keypoints_from_video(video_path, landmarker):
    """
    从单个视频提取关键点序列
    返回: (frames, 17, 3) 坐标数组, (frames, 17) 可见度数组
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [警告] 无法打开视频: {video_path}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # 默认

    all_coords = []
    all_vis = []
    prev_smoothed = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 创建 MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = landmarker.detect(mp_image)
        except Exception:
            frame_idx += 1
            if len(all_coords) > 0:
                all_coords.append(all_coords[-1].copy())
                all_vis.append(all_vis[-1].copy())
            continue

        coords, vis = extract_landmarks_from_result(result)

        if coords is not None:
            # 与推理路径保持一致的预处理：
            # 1) Pose 级健壮性检查：整帧躯干可见度/身体跨度崩坏或 hip 中心跳变过大时，
            #    直接复用上一帧平滑后的姿态，避免坏帧污染训练样本
            # 2) 关键点级可见度过滤与跳变插值
            # 3) 指数滑动平均平滑
            fallback_coords = prev_smoothed if prev_smoothed is not None else (
                all_coords[-1] if len(all_coords) > 0 else None
            )
            if should_reuse_previous_pose(coords, vis, prev_coords=fallback_coords):
                coords = fallback_coords.copy()
                vis = np.maximum(vis, VISIBILITY_THRESHOLD)
            coords = interpolate_missing(coords, vis, prev_coords=fallback_coords)

            # 首帧核心关节不可靠时 interpolate_missing 返回 None
            # 训练数据宁缺毋滥：直接跳过这一帧，等下一帧建立基准
            if coords is None:
                frame_idx += 1
                continue

            smoothed = smooth_keypoints(coords, prev_smoothed)
            prev_smoothed = smoothed

            all_coords.append(smoothed)
            all_vis.append(vis)
        else:
            # 无法检测到人体，使用上一帧数据
            if len(all_coords) > 0:
                all_coords.append(all_coords[-1].copy())
                all_vis.append(all_vis[-1].copy())

        frame_idx += 1

    cap.release()

    if len(all_coords) < WINDOW_SIZE:
        print(f"  [警告] 视频帧数不足 {WINDOW_SIZE}: {video_path} (仅 {len(all_coords)} 帧)")
        return None, None

    return np.array(all_coords), np.array(all_vis)


def create_sliding_windows(coords_seq, window_size=WINDOW_SIZE, step=WINDOW_STEP):
    """
    使用滑动窗口切分时序数据
    coords_seq: (frames, 17, 3)
    返回: list of (window_size, feature_dim) 特征窗口
    """
    windows = []
    num_frames = len(coords_seq)

    for start in range(0, num_frames - window_size + 1, step):
        window_coords = coords_seq[start:start + window_size]

        # 对窗口内每帧进行归一化和特征构建
        features = []
        for frame_coords in window_coords:
            normed = normalize_keypoints(frame_coords)
            feat = build_frame_feature(normed)
            features.append(feat)

        windows.append(np.array(features))

    return windows


# ==================== 数据增强 ====================

def augment_mirror(coords_seq):
    """镜像翻转"""
    mirrored = coords_seq.copy()
    mirrored[:, :, 0] = 1.0 - mirrored[:, :, 0]

    swap_pairs = [
        (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
        (KP_LEFT_ELBOW, KP_RIGHT_ELBOW),
        (KP_LEFT_WRIST, KP_RIGHT_WRIST),
        (KP_LEFT_HIP, KP_RIGHT_HIP),
        (KP_LEFT_KNEE, KP_RIGHT_KNEE),
        (KP_LEFT_ANKLE, KP_RIGHT_ANKLE),
        (KP_LEFT_HEEL, KP_RIGHT_HEEL),
        (KP_LEFT_FOOT, KP_RIGHT_FOOT),
    ]
    for (l, r) in swap_pairs:
        mirrored[:, [l, r]] = mirrored[:, [r, l]]

    return mirrored


def augment_noise(coords_seq, noise_std=0.005):
    """坐标噪声扰动"""
    noise = np.random.normal(0, noise_std, coords_seq.shape).astype(np.float32)
    return coords_seq + noise


def augment_drop_frames(coords_seq, drop_rate=0.1):
    """随机丢帧"""
    num_frames = len(coords_seq)
    result = coords_seq.copy()
    num_drop = int(num_frames * drop_rate)

    drop_indices = np.random.choice(
        range(1, num_frames - 1), 
        size=min(num_drop, num_frames - 2), 
        replace=False
    )
    for idx in drop_indices:
        result[idx] = (result[idx - 1] + result[min(idx + 1, num_frames - 1)]) / 2.0

    return result


def augment_time_scale(coords_seq, scale_range=(0.8, 1.2)):
    """时间尺度拉伸"""
    scale = np.random.uniform(*scale_range)
    num_frames = len(coords_seq)
    new_length = int(num_frames * scale)
    if new_length < WINDOW_SIZE:
        new_length = WINDOW_SIZE

    indices = np.linspace(0, num_frames - 1, new_length)
    result = np.zeros((new_length, coords_seq.shape[1], coords_seq.shape[2]), dtype=np.float32)

    for i, idx in enumerate(indices):
        low = int(idx)
        high = min(low + 1, num_frames - 1)
        frac = idx - low
        result[i] = coords_seq[low] * (1 - frac) + coords_seq[high] * frac

    return result


def apply_augmentations(coords_seq):
    """对单个视频坐标序列应用多种增强"""
    augmented = []
    augmented.append(augment_mirror(coords_seq))
    augmented.append(augment_noise(coords_seq, noise_std=0.003))
    augmented.append(augment_noise(coords_seq, noise_std=0.006))
    augmented.append(augment_drop_frames(coords_seq, drop_rate=0.1))
    augmented.append(augment_time_scale(coords_seq, scale_range=(0.85, 0.95)))
    augmented.append(augment_time_scale(coords_seq, scale_range=(1.05, 1.15)))
    mirror = augment_mirror(coords_seq)
    augmented.append(augment_noise(mirror, noise_std=0.004))
    return augmented


def process_split(split_name, material_dir, output_dir, landmarker, do_augment=False):
    """处理一个数据集分割"""
    split_dir = os.path.join(material_dir, split_name)
    out_dir = os.path.join(output_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    # 按类别分别收集样本，便于后续均衡截断
    class_samples = {}  # action -> list of sample arrays
    stats = {}

    for action in ACTION_CLASSES:
        action_dir = os.path.join(split_dir, action)
        if not os.path.exists(action_dir):
            print(f"[警告] 目录不存在: {action_dir}")
            continue

        videos = sorted([f for f in os.listdir(action_dir) if f.endswith(('.mp4', '.avi', '.mov'))])
        action_sample_list = []

        print(f"\n  处理 {split_name}/{action} ({len(videos)} 个视频)...")

        for video_file in tqdm(videos, desc=f"    {action}"):
            video_path = os.path.join(action_dir, video_file)
            coords, vis = extract_keypoints_from_video(video_path, landmarker)

            if coords is None:
                continue

            # 原始数据的滑动窗口
            windows = create_sliding_windows(coords)
            action_sample_list.extend(windows)

            # 训练集应用数据增强
            if do_augment:
                aug_versions = apply_augmentations(coords)
                for aug_coords in aug_versions:
                    aug_windows = create_sliding_windows(aug_coords)
                    action_sample_list.extend(aug_windows)

        class_samples[action] = action_sample_list
        print(f"    {ACTION_NAMES_CN[action]}: {len(action_sample_list)} 个样本")

    # 训练集类别均衡截断：超出上限的类别随机采样至上限
    if do_augment and MAX_TRAIN_SAMPLES_PER_CLASS > 0:
        for action, samples in class_samples.items():
            if len(samples) > MAX_TRAIN_SAMPLES_PER_CLASS:
                original_count = len(samples)
                indices = np.random.default_rng(42).choice(
                    len(samples), size=MAX_TRAIN_SAMPLES_PER_CLASS, replace=False
                )
                class_samples[action] = [samples[i] for i in sorted(indices)]
                print(f"    [均衡] {ACTION_NAMES_CN[action]}: {original_count} -> {MAX_TRAIN_SAMPLES_PER_CLASS} (随机截断)")

    # 合并所有类别
    all_samples = []
    all_labels = []
    for action, samples in class_samples.items():
        label_idx = CLASS_TO_IDX[action]
        all_samples.extend(samples)
        all_labels.extend([label_idx] * len(samples))
        stats[action] = len(samples)

    if len(all_samples) == 0:
        print(f"[错误] {split_name} 没有生成任何样本！")
        return stats

    samples_array = np.array(all_samples, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    np.save(os.path.join(out_dir, "samples.npy"), samples_array)
    np.save(os.path.join(out_dir, "labels.npy"), labels_array)

    print(f"\n  {split_name} 总计: {len(all_samples)} 个样本")
    print(f"  样本形状: {samples_array.shape}")
    print(f"  保存到: {out_dir}")

    return stats


def main():
    """主程序入口"""
    print("=" * 60)
    print("  关键点提取与样本生成")
    print("=" * 60)

    # 初始化 PoseLandmarker
    model_path = os.path.join(MODEL_DIR, "pose_landmarker_heavy.task")
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先下载: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        return

    landmarker = create_pose_landmarker(model_path)

    output_dir = DATA_DIR
    all_stats = {}

    # 处理训练集（含数据增强）
    print("\n" + "=" * 40)
    print("  处理训练集 (含数据增强)")
    print("=" * 40)
    all_stats["train"] = process_split("train", MATERIAL_DIR, output_dir, landmarker, do_augment=True)

    # 处理验证集
    print("\n" + "=" * 40)
    print("  处理验证集")
    print("=" * 40)
    all_stats["verify"] = process_split("verify", MATERIAL_DIR, output_dir, landmarker, do_augment=False)

    # 处理测试集
    print("\n" + "=" * 40)
    print("  处理测试集")
    print("=" * 40)
    all_stats["test"] = process_split("test", MATERIAL_DIR, output_dir, landmarker, do_augment=False)

    landmarker.close()

    # 保存统计信息
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  数据集构建完成！")
    print("=" * 60)
    print(f"\n统计信息已保存到: {stats_path}")

    for split in ["train", "verify", "test"]:
        if split in all_stats:
            total = sum(all_stats[split].values())
            print(f"\n{split}: {total} 个样本")
            for action, count in all_stats[split].items():
                print(f"  {ACTION_NAMES_CN.get(action, action)}: {count}")


if __name__ == "__main__":
    main()
