# -*- coding: utf-8 -*-
"""
动作计数模块
基于有限状态机的动作自动计次

设计：
1. ACTIVE 状态超时重置 —— 防止卡在 ACTIVE 导致后续动作全部漏计
2. 放宽状态确认帧数 —— 适配隔帧检测模式
3. 统一四种动作的超时机制
4. 弓步蹲角度 EMA 平滑 + 异常值过滤 —— 解决关键点低可见度导致的骨骼跳变
"""
import numpy as np
from config import *
from utils import calculate_angle


class ActionState:
    """动作状态枚举"""
    READY = "ready"       # 准备态
    ACTIVE = "active"     # 动作执行态
    COMPLETE = "complete" # 完成态


# ACTIVE 状态最大持续帧数，超过则认为状态机卡死，强制重置到 READY
# 30fps 下 90 帧 = 3 秒，一个正常动作不应超过 3 秒
ACTIVE_TIMEOUT_FRAMES = 90

# 角度 EMA 平滑系数（值越大越跟随当前帧，越小越平滑）
ANGLE_EMA_ALPHA = 0.5
# 快速动作使用较弱的 EMA，避免峰值被削弱（影响最小/最大角度捕捉）
ANGLE_EMA_ALPHA_FAST = 0.85
# 膝角异常值范围：正常人膝角不可能低于30°或高于180°
KNEE_ANGLE_MIN = 30.0
KNEE_ANGLE_MAX = 178.0
# 肘角异常值范围：俯卧撑肘角不可能低于30°
ELBOW_ANGLE_MIN = 30.0
ELBOW_ANGLE_MAX = 178.0
# 躯干角异常值范围：卷腹/深蹲躯干角不可能超出 [0°, 180°]，取保守上下限
TORSO_ANGLE_MIN = 0.0
TORSO_ANGLE_MAX = 180.0
# 帧间膝角最大允许变化量（超过此值视为跳变，沿用上一帧）
MAX_ANGLE_JUMP = 40.0


class ActionCounter:
    """
    基于有限状态机的动作计数器
    每种动作定义不同的状态转移条件
    """

    def __init__(self):
        self.state = ActionState.READY
        self.count = 0
        self.state_frames = 0  # 当前状态持续帧数
        self.current_action = None
        self.angle_history = []  # 角度历史（用于评分）

        # 每次动作的角度记录
        self.current_rep_angles = {}

        # 弓步蹲前腿追踪：锁定单次动作中的前腿侧
        self._lunge_front_side = None  # "left" or "right"

        # 记录进入 ACTIVE 时的角度，用于判断是否真正完成了动作
        self._active_entry_angle = None

        # 自动识别模式抗抖动：需连续多帧识别为不同动作才切换
        self._pending_action = None
        self._pending_action_frames = 0

        # 深蹲角度 EMA 平滑缓存
        self._squat_smooth_left_knee = None
        self._squat_smooth_right_knee = None
        self._squat_smooth_torso = None

        # 俯卧撑角度 EMA 平滑缓存（使用较弱 alpha 避免峰值削弱）
        self._pushup_smooth_left_elbow = None
        self._pushup_smooth_right_elbow = None
        self._pushup_smooth_body = None

        # 卷腹角度 EMA 平滑缓存
        self._crunch_smooth_torso = None

        # 弓步蹲角度 EMA 平滑缓存
        self._lunge_smooth_left_knee = None
        self._lunge_smooth_right_knee = None
        self._lunge_smooth_torso = None

    def reset(self, action=None):
        """重置计数器"""
        self.state = ActionState.READY
        self.count = 0
        self.state_frames = 0
        self.current_action = action
        self.angle_history = []
        self.current_rep_angles = {}
        self._lunge_front_side = None
        self._active_entry_angle = None
        self._pending_action = None
        self._pending_action_frames = 0
        self._squat_smooth_left_knee = None
        self._squat_smooth_right_knee = None
        self._squat_smooth_torso = None
        self._pushup_smooth_left_elbow = None
        self._pushup_smooth_right_elbow = None
        self._pushup_smooth_body = None
        self._crunch_smooth_torso = None
        self._lunge_smooth_left_knee = None
        self._lunge_smooth_right_knee = None
        self._lunge_smooth_torso = None

    # 动作切换需要连续确认的帧数
    ACTION_SWITCH_CONFIRM = 10

    def update(self, coords_17, action_key):
        """
        更新状态机
        coords_17: (17, 3) 原始坐标（非归一化，用于角度计算）
        action_key: 动作类型 key (squat/push-up/crunches/lunge)
        返回: (是否新增计数, 当前关键角度字典)
        """
        if action_key != self.current_action:
            # 抗抖动：自动识别模式下，需连续多帧识别为新动作才真正切换
            # 防止偶尔的误分类导致计数器归零
            if self.current_action is None:
                # 首次设置，直接生效
                self.reset(action_key)
            elif action_key == self._pending_action:
                self._pending_action_frames += 1
                if self._pending_action_frames >= self.ACTION_SWITCH_CONFIRM:
                    # 连续确认足够多帧，执行切换（保留count）
                    old_count = self.count
                    old_history = self.angle_history
                    self.reset(action_key)
                    self.count = old_count
                    self.angle_history = old_history
            else:
                # 新的候选动作，开始计数
                self._pending_action = action_key
                self._pending_action_frames = 1
            # 在切换确认期间，继续用当前动作处理
            if action_key != self.current_action:
                action_key = self.current_action
                if action_key is None:
                    return False, {}
        else:
            # 动作一致，清空切换候选
            self._pending_action = None
            self._pending_action_frames = 0

        if action_key == "squat":
            return self._update_squat(coords_17)
        elif action_key == "push-up":
            return self._update_pushup(coords_17)
        elif action_key == "crunches":
            return self._update_crunches(coords_17)
        elif action_key == "lunge":
            return self._update_lunge(coords_17)

        return False, {}

    def _check_active_timeout(self):
        """
        检查 ACTIVE 状态是否超时
        如果在 ACTIVE 停留过久，说明状态机卡死了（可能膝角一直无法达到 UP 阈值），
        强制重置到 READY，避免后续所有动作都被漏计
        """
        if self.state == ActionState.ACTIVE and self.state_frames > ACTIVE_TIMEOUT_FRAMES:
            self.state = ActionState.READY
            self.state_frames = 0
            self.current_rep_angles = {}
            self._active_entry_angle = None
            return True  # 发生了超时重置
        return False

    def _update_squat(self, coords):
        """深蹲状态机"""
        # 计算左右膝角（含异常过滤 + 跳变抑制 + EMA）
        raw_left_knee = calculate_angle(
            coords[KP_LEFT_HIP], coords[KP_LEFT_KNEE], coords[KP_LEFT_ANKLE]
        )
        raw_right_knee = calculate_angle(
            coords[KP_RIGHT_HIP], coords[KP_RIGHT_KNEE], coords[KP_RIGHT_ANKLE]
        )
        left_knee_angle = self._sanitize_angle(
            raw_left_knee, self._squat_smooth_left_knee,
            lo=KNEE_ANGLE_MIN, hi=KNEE_ANGLE_MAX, fallback=160.0
        )
        right_knee_angle = self._sanitize_angle(
            raw_right_knee, self._squat_smooth_right_knee,
            lo=KNEE_ANGLE_MIN, hi=KNEE_ANGLE_MAX, fallback=160.0
        )
        self._squat_smooth_left_knee = left_knee_angle
        self._squat_smooth_right_knee = right_knee_angle
        knee_angle = (left_knee_angle + right_knee_angle) / 2.0

        # 躯干倾角（做异常过滤 + 平滑）
        shoulder_center = (coords[KP_LEFT_SHOULDER] + coords[KP_RIGHT_SHOULDER]) / 2.0
        hip_center = (coords[KP_LEFT_HIP] + coords[KP_RIGHT_HIP]) / 2.0
        vertical = np.array([0, -1, 0])
        raw_torso_angle = calculate_angle(shoulder_center, hip_center, hip_center + vertical)
        torso_angle = self._sanitize_angle(
            raw_torso_angle, self._squat_smooth_torso,
            lo=TORSO_ANGLE_MIN, hi=TORSO_ANGLE_MAX, fallback=10.0
        )
        self._squat_smooth_torso = torso_angle

        angles = {
            "knee_angle": knee_angle,
            "torso_angle": torso_angle,
            "left_knee": left_knee_angle,
            "right_knee": right_knee_angle,
        }

        completed = False

        # 超时检查
        self._check_active_timeout()

        if self.state == ActionState.READY:
            if knee_angle < SQUAT_KNEE_ANGLE_DOWN:
                self.state = ActionState.ACTIVE
                self.state_frames = 0
                self._active_entry_angle = knee_angle
                self.current_rep_angles = {"min_knee": knee_angle, "max_torso_lean": torso_angle}

        elif self.state == ActionState.ACTIVE:
            self.state_frames += 1
            # 记录最小膝角和最大躯干倾斜
            if knee_angle < self.current_rep_angles.get("min_knee", 180):
                self.current_rep_angles["min_knee"] = knee_angle
            if torso_angle > self.current_rep_angles.get("max_torso_lean", 0):
                self.current_rep_angles["max_torso_lean"] = torso_angle

            if knee_angle > SQUAT_KNEE_ANGLE_UP and self.state_frames >= SQUAT_MIN_ACTIVE_FRAMES:
                # 最小动作幅度校验：站起膝角与最低膝角的差值需超过阈值
                # 防止角度抖动导致的假计数
                min_knee = self.current_rep_angles.get("min_knee", knee_angle)
                if (knee_angle - min_knee) >= SQUAT_MIN_ROM:
                    self.state = ActionState.COMPLETE
                    self.state_frames = 0

        elif self.state == ActionState.COMPLETE:
            self.count += 1
            completed = True
            self.angle_history.append(self.current_rep_angles.copy())
            self.state = ActionState.READY
            self.state_frames = 0
            self.current_rep_angles = {}
            self._active_entry_angle = None

        return completed, angles

    def _update_pushup(self, coords):
        """俯卧撑状态机"""
        # 左右肘角（俯卧撑动作快，用 ALPHA_FAST 较弱平滑避免峰值削弱，
        # 但保留异常过滤 + 跳变抑制防止单帧坐标错误误触发）
        raw_left_elbow = calculate_angle(
            coords[KP_LEFT_SHOULDER], coords[KP_LEFT_ELBOW], coords[KP_LEFT_WRIST]
        )
        raw_right_elbow = calculate_angle(
            coords[KP_RIGHT_SHOULDER], coords[KP_RIGHT_ELBOW], coords[KP_RIGHT_WRIST]
        )
        left_elbow_angle = self._sanitize_angle(
            raw_left_elbow, self._pushup_smooth_left_elbow,
            lo=ELBOW_ANGLE_MIN, hi=ELBOW_ANGLE_MAX,
            alpha=ANGLE_EMA_ALPHA_FAST, fallback=160.0
        )
        right_elbow_angle = self._sanitize_angle(
            raw_right_elbow, self._pushup_smooth_right_elbow,
            lo=ELBOW_ANGLE_MIN, hi=ELBOW_ANGLE_MAX,
            alpha=ANGLE_EMA_ALPHA_FAST, fallback=160.0
        )
        self._pushup_smooth_left_elbow = left_elbow_angle
        self._pushup_smooth_right_elbow = right_elbow_angle
        elbow_angle = (left_elbow_angle + right_elbow_angle) / 2.0

        # 肩髋踝连线角度（body straightness，做异常过滤 + 轻平滑）
        raw_body_angle = calculate_angle(
            coords[KP_LEFT_SHOULDER], coords[KP_LEFT_HIP], coords[KP_LEFT_ANKLE]
        )
        body_angle = self._sanitize_angle(
            raw_body_angle, self._pushup_smooth_body,
            lo=TORSO_ANGLE_MIN, hi=TORSO_ANGLE_MAX,
            alpha=ANGLE_EMA_ALPHA_FAST, fallback=175.0
        )
        self._pushup_smooth_body = body_angle

        angles = {
            "elbow_angle": elbow_angle,
            "body_angle": body_angle,
            "left_elbow": left_elbow_angle,
            "right_elbow": right_elbow_angle,
        }

        completed = False

        # 超时检查
        self._check_active_timeout()

        if self.state == ActionState.READY:
            if elbow_angle < PUSHUP_ELBOW_ANGLE_DOWN:
                self.state = ActionState.ACTIVE
                self.state_frames = 0
                self.current_rep_angles = {"min_elbow": elbow_angle, "body_bend": abs(180 - body_angle)}

        elif self.state == ActionState.ACTIVE:
            self.state_frames += 1
            if elbow_angle < self.current_rep_angles.get("min_elbow", 180):
                self.current_rep_angles["min_elbow"] = elbow_angle
            body_bend = abs(180 - body_angle)
            if body_bend > self.current_rep_angles.get("body_bend", 0):
                self.current_rep_angles["body_bend"] = body_bend

            if elbow_angle > PUSHUP_ELBOW_ANGLE_UP and self.state_frames >= PUSHUP_MIN_ACTIVE_FRAMES:
                # 最小动作幅度校验：肘角从最低点到当前值的变化需超过阈值
                # 防止噪声或外推过冲导致的虚假计数
                min_elbow = self.current_rep_angles.get("min_elbow", elbow_angle)
                if (elbow_angle - min_elbow) >= PUSHUP_MIN_ROM:
                    self.state = ActionState.COMPLETE
                    self.state_frames = 0

        elif self.state == ActionState.COMPLETE:
            self.count += 1
            completed = True
            self.angle_history.append(self.current_rep_angles.copy())
            self.state = ActionState.READY
            self.state_frames = 0
            self.current_rep_angles = {}

        return completed, angles

    def _update_crunches(self, coords):
        """卷腹状态机"""
        # 躯干与大腿夹角（做异常过滤 + EMA 平滑）
        shoulder_center = (coords[KP_LEFT_SHOULDER] + coords[KP_RIGHT_SHOULDER]) / 2.0
        hip_center = (coords[KP_LEFT_HIP] + coords[KP_RIGHT_HIP]) / 2.0
        knee_center = (coords[KP_LEFT_KNEE] + coords[KP_RIGHT_KNEE]) / 2.0

        raw_torso_angle = calculate_angle(shoulder_center, hip_center, knee_center)
        torso_angle = self._sanitize_angle(
            raw_torso_angle, self._crunch_smooth_torso,
            lo=TORSO_ANGLE_MIN, hi=TORSO_ANGLE_MAX, fallback=160.0
        )
        self._crunch_smooth_torso = torso_angle

        angles = {
            "torso_angle": torso_angle,
        }

        completed = False

        # 超时检查
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
                self.state = ActionState.COMPLETE
                self.state_frames = 0

        elif self.state == ActionState.COMPLETE:
            self.count += 1
            completed = True
            self.angle_history.append(self.current_rep_angles.copy())
            self.state = ActionState.READY
            self.state_frames = 0
            self.current_rep_angles = {}

        return completed, angles

    def _sanitize_angle(self, raw, prev_smooth,
                        lo=KNEE_ANGLE_MIN, hi=KNEE_ANGLE_MAX,
                        alpha=ANGLE_EMA_ALPHA, fallback=None):
        """
        对角度做"异常值过滤 + 跳变抑制 + EMA 平滑"三重保护
        - 超出 [lo, hi] 的物理不可能值视为坐标错误，沿用上一帧（首帧用 fallback）
        - 帧间变化超过 MAX_ANGLE_JUMP 视为跳变，沿用上一帧
        - 通过以上两个闸门后再做 EMA 平滑

        alpha 越大越贴合当前帧（alpha=1 表示不做 EMA）
        """
        if not (lo <= raw <= hi):
            raw = prev_smooth if prev_smooth is not None else (
                fallback if fallback is not None else raw
            )
        if prev_smooth is None:
            return raw
        if abs(raw - prev_smooth) > MAX_ANGLE_JUMP:
            return prev_smooth
        return alpha * raw + (1 - alpha) * prev_smooth

    # 向后兼容：保留旧名称
    def _smooth_angle(self, raw, prev_smooth, alpha=ANGLE_EMA_ALPHA):
        return self._sanitize_angle(raw, prev_smooth, alpha=alpha)

    def _update_lunge(self, coords):
        """
        弓步蹲状态机
        改进：
        1. 膝角 EMA 平滑 + 跳变抑制 —— 解决低可见度关键点导致角度剧烈波动
        2. 异常值过滤 —— 膝角 < 30° 视为坐标错误，沿用上一帧
        3. 前腿锁定更稳定 —— 需要连续多帧一致判定才更新
        """
        # 计算左右膝角（异常过滤 + 跳变抑制 + EMA）
        raw_left_knee = calculate_angle(
            coords[KP_LEFT_HIP], coords[KP_LEFT_KNEE], coords[KP_LEFT_ANKLE]
        )
        raw_right_knee = calculate_angle(
            coords[KP_RIGHT_HIP], coords[KP_RIGHT_KNEE], coords[KP_RIGHT_ANKLE]
        )
        left_knee_angle = self._sanitize_angle(
            raw_left_knee, self._lunge_smooth_left_knee,
            lo=KNEE_ANGLE_MIN, hi=KNEE_ANGLE_MAX, fallback=160.0
        )
        right_knee_angle = self._sanitize_angle(
            raw_right_knee, self._lunge_smooth_right_knee,
            lo=KNEE_ANGLE_MIN, hi=KNEE_ANGLE_MAX, fallback=160.0
        )
        self._lunge_smooth_left_knee = left_knee_angle
        self._lunge_smooth_right_knee = right_knee_angle

        # 前腿判定：弓步蹲时前腿弯曲更深、后腿伸直，膝角更小的一侧即为前腿
        # 在 READY 状态下不断更新，进入 ACTIVE 后锁定，直到该次动作完成
        if self._lunge_front_side is None or self.state == ActionState.READY:
            # 两腿膝角差异足够大时才更新判定（避免站立时微小差异导致误判）
            angle_diff = abs(left_knee_angle - right_knee_angle)
            if angle_diff > 20:
                if left_knee_angle < right_knee_angle:
                    self._lunge_front_side = "left"
                else:
                    self._lunge_front_side = "right"

        if self._lunge_front_side == "left":
            front_knee_angle = left_knee_angle
            back_knee_angle = right_knee_angle
        else:
            front_knee_angle = right_knee_angle
            back_knee_angle = left_knee_angle

        # 躯干直立度（异常过滤 + 跳变抑制 + EMA 平滑）
        # 弓步蹲应保持近乎直立，躯干偏离垂直线 > 80° 明显为坐标错误
        shoulder_center = (coords[KP_LEFT_SHOULDER] + coords[KP_RIGHT_SHOULDER]) / 2.0
        hip_center = (coords[KP_LEFT_HIP] + coords[KP_RIGHT_HIP]) / 2.0
        vertical = np.array([0, -1, 0])
        raw_torso_lean = calculate_angle(shoulder_center, hip_center, hip_center + vertical)
        torso_lean = self._sanitize_angle(
            raw_torso_lean, self._lunge_smooth_torso,
            lo=0.0, hi=80.0, fallback=10.0
        )
        self._lunge_smooth_torso = torso_lean

        angles = {
            "front_knee_angle": front_knee_angle,
            "back_knee_angle": back_knee_angle,
            "torso_lean": torso_lean,
            "left_knee": left_knee_angle,
            "right_knee": right_knee_angle,
        }

        completed = False

        # 超时检查
        self._check_active_timeout()

        if self.state == ActionState.READY:
            if front_knee_angle < LUNGE_KNEE_ANGLE_DOWN:
                self.state = ActionState.ACTIVE
                self.state_frames = 0
                self.current_rep_angles = {
                    "min_front_knee": front_knee_angle,
                    "max_torso_lean": torso_lean
                }

        elif self.state == ActionState.ACTIVE:
            self.state_frames += 1
            if front_knee_angle < self.current_rep_angles.get("min_front_knee", 180):
                self.current_rep_angles["min_front_knee"] = front_knee_angle
            if torso_lean > self.current_rep_angles.get("max_torso_lean", 0):
                self.current_rep_angles["max_torso_lean"] = torso_lean

            avg_knee = (left_knee_angle + right_knee_angle) / 2.0
            if avg_knee > LUNGE_KNEE_ANGLE_UP and self.state_frames >= STATE_CONFIRM_FRAMES:
                self.state = ActionState.COMPLETE
                self.state_frames = 0

        elif self.state == ActionState.COMPLETE:
            self.count += 1
            completed = True
            self.angle_history.append(self.current_rep_angles.copy())
            # 完成一次后释放前腿锁定，让下一次重新判定
            self._lunge_front_side = None
            self.state = ActionState.READY
            self.state_frames = 0
            self.current_rep_angles = {}

        return completed, angles
