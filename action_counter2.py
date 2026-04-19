# -*- coding: utf-8 -*-
"""
改进版动作计数模块

修复点：
1. 达到完成条件时立即计数，避免视频在完成帧结束时漏掉最后一次。
2. 自动识别动作切换确认期间暂停计数，避免把新动作姿态继续按旧动作处理。
3. 修正弓步蹲前腿判定，去掉“默认右腿”偏置，并用角度 + y 坐标联合确认。
4. 弓步蹲完成条件改为以前腿恢复为主，并补充最小动作幅度校验。
5. 卷腹补充最小动作幅度校验，降低阈值抖动带来的误计数。
"""

import numpy as np

from config import *
from utils import calculate_angle
from action_counter import (
    ActionCounter as LegacyActionCounter,
    ActionState,
    ANGLE_EMA_ALPHA,
    KNEE_ANGLE_MIN,
    MAX_ANGLE_JUMP,
)


LUNGE_FRONT_CONFIRM_FRAMES = 3
LUNGE_FRONT_ANGLE_DIFF = 10.0
LUNGE_FRONT_Y_DIFF = 0.02
LUNGE_MIN_ROM = 15.0
CRUNCH_MIN_ROM = 15.0


class ActionCounter(LegacyActionCounter):
    """
    与原版保持相同接口的改进计数器。
    """

    def __init__(self):
        super().__init__()
        self._lunge_front_candidate = None
        self._lunge_front_candidate_frames = 0

    def reset(self, action=None):
        super().reset(action)
        self._lunge_front_candidate = None
        self._lunge_front_candidate_frames = 0

    def update(self, coords_17, action_key):
        """
        动作切换确认期间不再把当前帧继续喂给旧动作状态机，
        避免自动识别模式下在动作边界处误计数。
        """
        if action_key != self.current_action:
            if self.current_action is None:
                self.reset(action_key)
            elif action_key == self._pending_action:
                self._pending_action_frames += 1
                if self._pending_action_frames >= self.ACTION_SWITCH_CONFIRM:
                    old_count = self.count
                    old_history = list(self.angle_history)
                    self.reset(action_key)
                    self.count = old_count
                    self.angle_history = old_history
                else:
                    return False, {}
            else:
                self._pending_action = action_key
                self._pending_action_frames = 1
                return False, {}
        else:
            self._pending_action = None
            self._pending_action_frames = 0

        if action_key == "squat":
            return self._update_squat(coords_17)
        if action_key == "push-up":
            return self._update_pushup(coords_17)
        if action_key == "crunches":
            return self._update_crunches(coords_17)
        if action_key == "lunge":
            return self._update_lunge(coords_17)

        return False, {}

    def _check_active_timeout(self):
        timed_out = super()._check_active_timeout()
        if timed_out:
            self._reset_lunge_front_tracking()
        return timed_out

    def _finish_rep(self, reset_lunge=False):
        if self.current_rep_angles:
            self.angle_history.append(self.current_rep_angles.copy())
        self.count += 1
        self.state = ActionState.READY
        self.state_frames = 0
        self.current_rep_angles = {}
        self._active_entry_angle = None
        if reset_lunge:
            self._reset_lunge_front_tracking()
        return True

    def _reset_lunge_front_tracking(self):
        self._lunge_front_side = None
        self._lunge_front_candidate = None
        self._lunge_front_candidate_frames = 0

    def _smooth_angle(self, raw, prev_smooth, alpha=ANGLE_EMA_ALPHA):
        if prev_smooth is None:
            return raw
        if abs(raw - prev_smooth) > MAX_ANGLE_JUMP:
            return prev_smooth
        return alpha * raw + (1 - alpha) * prev_smooth

    def _update_squat(self, coords):
        left_knee_angle = calculate_angle(
            coords[KP_LEFT_HIP], coords[KP_LEFT_KNEE], coords[KP_LEFT_ANKLE]
        )
        right_knee_angle = calculate_angle(
            coords[KP_RIGHT_HIP], coords[KP_RIGHT_KNEE], coords[KP_RIGHT_ANKLE]
        )
        knee_angle = (left_knee_angle + right_knee_angle) / 2.0

        shoulder_center = (coords[KP_LEFT_SHOULDER] + coords[KP_RIGHT_SHOULDER]) / 2.0
        hip_center = (coords[KP_LEFT_HIP] + coords[KP_RIGHT_HIP]) / 2.0
        vertical = np.array([0, -1, 0])
        torso_angle = calculate_angle(shoulder_center, hip_center, hip_center + vertical)

        angles = {
            "knee_angle": knee_angle,
            "torso_angle": torso_angle,
            "left_knee": left_knee_angle,
            "right_knee": right_knee_angle,
        }

        completed = False
        self._check_active_timeout()

        if self.state == ActionState.READY:
            if knee_angle < SQUAT_KNEE_ANGLE_DOWN:
                self.state = ActionState.ACTIVE
                self.state_frames = 0
                self._active_entry_angle = knee_angle
                self.current_rep_angles = {
                    "min_knee": knee_angle,
                    "max_torso_lean": torso_angle,
                }

        elif self.state == ActionState.ACTIVE:
            self.state_frames += 1
            if knee_angle < self.current_rep_angles.get("min_knee", 180):
                self.current_rep_angles["min_knee"] = knee_angle
            if torso_angle > self.current_rep_angles.get("max_torso_lean", 0):
                self.current_rep_angles["max_torso_lean"] = torso_angle

            if knee_angle > SQUAT_KNEE_ANGLE_UP and self.state_frames >= SQUAT_MIN_ACTIVE_FRAMES:
                min_knee = self.current_rep_angles.get("min_knee", knee_angle)
                if (knee_angle - min_knee) >= SQUAT_MIN_ROM:
                    completed = self._finish_rep()

        return completed, angles

    def _update_pushup(self, coords):
        left_elbow_angle = calculate_angle(
            coords[KP_LEFT_SHOULDER], coords[KP_LEFT_ELBOW], coords[KP_LEFT_WRIST]
        )
        right_elbow_angle = calculate_angle(
            coords[KP_RIGHT_SHOULDER], coords[KP_RIGHT_ELBOW], coords[KP_RIGHT_WRIST]
        )
        elbow_angle = (left_elbow_angle + right_elbow_angle) / 2.0

        body_angle = calculate_angle(
            coords[KP_LEFT_SHOULDER], coords[KP_LEFT_HIP], coords[KP_LEFT_ANKLE]
        )

        angles = {
            "elbow_angle": elbow_angle,
            "body_angle": body_angle,
            "left_elbow": left_elbow_angle,
            "right_elbow": right_elbow_angle,
        }

        completed = False
        self._check_active_timeout()

        if self.state == ActionState.READY:
            if elbow_angle < PUSHUP_ELBOW_ANGLE_DOWN:
                self.state = ActionState.ACTIVE
                self.state_frames = 0
                self.current_rep_angles = {
                    "min_elbow": elbow_angle,
                    "body_bend": abs(180 - body_angle),
                }

        elif self.state == ActionState.ACTIVE:
            self.state_frames += 1
            if elbow_angle < self.current_rep_angles.get("min_elbow", 180):
                self.current_rep_angles["min_elbow"] = elbow_angle
            body_bend = abs(180 - body_angle)
            if body_bend > self.current_rep_angles.get("body_bend", 0):
                self.current_rep_angles["body_bend"] = body_bend

            if elbow_angle > PUSHUP_ELBOW_ANGLE_UP and self.state_frames >= PUSHUP_MIN_ACTIVE_FRAMES:
                min_elbow = self.current_rep_angles.get("min_elbow", elbow_angle)
                if (elbow_angle - min_elbow) >= PUSHUP_MIN_ROM:
                    completed = self._finish_rep()

        return completed, angles

    def _update_crunches(self, coords):
        shoulder_center = (coords[KP_LEFT_SHOULDER] + coords[KP_RIGHT_SHOULDER]) / 2.0
        hip_center = (coords[KP_LEFT_HIP] + coords[KP_RIGHT_HIP]) / 2.0
        knee_center = (coords[KP_LEFT_KNEE] + coords[KP_RIGHT_KNEE]) / 2.0

        torso_angle = calculate_angle(shoulder_center, hip_center, knee_center)
        angles = {"torso_angle": torso_angle}

        completed = False
        self._check_active_timeout()

        if self.state == ActionState.READY:
            if torso_angle < CRUNCH_TORSO_ANGLE_UP:
                self.state = ActionState.ACTIVE
                self.state_frames = 0
                self.current_rep_angles = {"min_torso": torso_angle}

        elif self.state == ActionState.ACTIVE:
            self.state_frames += 1
            if torso_angle < self.current_rep_angles.get("min_torso", 180):
                self.current_rep_angles["min_torso"] = torso_angle

            if torso_angle > CRUNCH_TORSO_ANGLE_REST and self.state_frames >= STATE_CONFIRM_FRAMES:
                min_torso = self.current_rep_angles.get("min_torso", torso_angle)
                if (torso_angle - min_torso) >= CRUNCH_MIN_ROM:
                    completed = self._finish_rep()

        return completed, angles

    def _pick_lunge_front_candidate(
        self, left_knee_angle, right_knee_angle, left_knee_y, right_knee_y
    ):
        angle_diff = abs(left_knee_angle - right_knee_angle)
        if angle_diff >= LUNGE_FRONT_ANGLE_DIFF:
            return "left" if left_knee_angle < right_knee_angle else "right"

        y_diff = left_knee_y - right_knee_y
        if abs(y_diff) >= LUNGE_FRONT_Y_DIFF:
            return "left" if y_diff > 0 else "right"

        return None

    def _update_lunge_front_side(
        self, left_knee_angle, right_knee_angle, left_knee_y, right_knee_y
    ):
        candidate = self._pick_lunge_front_candidate(
            left_knee_angle, right_knee_angle, left_knee_y, right_knee_y
        )
        if candidate is None:
            self._lunge_front_candidate = None
            self._lunge_front_candidate_frames = 0
            return

        if candidate == self._lunge_front_candidate:
            self._lunge_front_candidate_frames += 1
        else:
            self._lunge_front_candidate = candidate
            self._lunge_front_candidate_frames = 1

        if self._lunge_front_candidate_frames >= LUNGE_FRONT_CONFIRM_FRAMES:
            self._lunge_front_side = candidate

    def _get_lunge_front_back_angles(self, left_knee_angle, right_knee_angle):
        if self._lunge_front_side == "left":
            return left_knee_angle, right_knee_angle
        if self._lunge_front_side == "right":
            return right_knee_angle, left_knee_angle
        if left_knee_angle <= right_knee_angle:
            return left_knee_angle, right_knee_angle
        return right_knee_angle, left_knee_angle

    def _update_lunge(self, coords):
        raw_left_knee = calculate_angle(
            coords[KP_LEFT_HIP], coords[KP_LEFT_KNEE], coords[KP_LEFT_ANKLE]
        )
        raw_right_knee = calculate_angle(
            coords[KP_RIGHT_HIP], coords[KP_RIGHT_KNEE], coords[KP_RIGHT_ANKLE]
        )

        if raw_left_knee < KNEE_ANGLE_MIN:
            raw_left_knee = (
                self._lunge_smooth_left_knee if self._lunge_smooth_left_knee is not None else 160.0
            )
        if raw_right_knee < KNEE_ANGLE_MIN:
            raw_right_knee = (
                self._lunge_smooth_right_knee if self._lunge_smooth_right_knee is not None else 160.0
            )

        left_knee_angle = self._smooth_angle(raw_left_knee, self._lunge_smooth_left_knee)
        right_knee_angle = self._smooth_angle(raw_right_knee, self._lunge_smooth_right_knee)
        self._lunge_smooth_left_knee = left_knee_angle
        self._lunge_smooth_right_knee = right_knee_angle

        left_knee_y = coords[KP_LEFT_KNEE][1]
        right_knee_y = coords[KP_RIGHT_KNEE][1]
        if self.state == ActionState.READY or self._lunge_front_side is None:
            self._update_lunge_front_side(
                left_knee_angle, right_knee_angle, left_knee_y, right_knee_y
            )

        front_knee_angle, back_knee_angle = self._get_lunge_front_back_angles(
            left_knee_angle, right_knee_angle
        )

        shoulder_center = (coords[KP_LEFT_SHOULDER] + coords[KP_RIGHT_SHOULDER]) / 2.0
        hip_center = (coords[KP_LEFT_HIP] + coords[KP_RIGHT_HIP]) / 2.0
        vertical = np.array([0, -1, 0])
        raw_torso_lean = calculate_angle(shoulder_center, hip_center, hip_center + vertical)
        if raw_torso_lean > 80:
            raw_torso_lean = self._lunge_smooth_torso if self._lunge_smooth_torso is not None else 10.0
        torso_lean = self._smooth_angle(raw_torso_lean, self._lunge_smooth_torso)
        self._lunge_smooth_torso = torso_lean

        angles = {
            "front_knee_angle": front_knee_angle,
            "back_knee_angle": back_knee_angle,
            "torso_lean": torso_lean,
            "left_knee": left_knee_angle,
            "right_knee": right_knee_angle,
        }

        completed = False
        self._check_active_timeout()

        if self.state == ActionState.READY:
            if self._lunge_front_side is None:
                self._lunge_front_side = "left" if left_knee_angle <= right_knee_angle else "right"
                front_knee_angle, back_knee_angle = self._get_lunge_front_back_angles(
                    left_knee_angle, right_knee_angle
                )

            if front_knee_angle < LUNGE_KNEE_ANGLE_DOWN:
                self.state = ActionState.ACTIVE
                self.state_frames = 0
                self.current_rep_angles = {
                    "min_front_knee": front_knee_angle,
                    "max_torso_lean": torso_lean,
                }

        elif self.state == ActionState.ACTIVE:
            self.state_frames += 1
            if front_knee_angle < self.current_rep_angles.get("min_front_knee", 180):
                self.current_rep_angles["min_front_knee"] = front_knee_angle
            if torso_lean > self.current_rep_angles.get("max_torso_lean", 0):
                self.current_rep_angles["max_torso_lean"] = torso_lean

            min_front_knee = self.current_rep_angles.get("min_front_knee", front_knee_angle)
            if front_knee_angle > LUNGE_KNEE_ANGLE_UP and self.state_frames >= STATE_CONFIRM_FRAMES:
                if (front_knee_angle - min_front_knee) >= LUNGE_MIN_ROM:
                    completed = self._finish_rep(reset_lunge=True)

        return completed, angles
