# -*- coding: utf-8 -*-
"""
动作规范性评分模块
S_total = 0.5 × S_angle + 0.3 × S_range + 0.2 × S_stable

改进：使用线性插值评分替代硬阈值台阶，让不同质量的动作获得不同分数
"""
import numpy as np
from config import *


def _linear_score(value, best, good, bad, higher_is_better=True):
    """
    线性插值评分工具
    value: 实际值
    best: 满分对应的值（得100分）
    good: 良好对应的值（得80分）
    bad:  最低对应的值（得30分）
    higher_is_better: True表示值越大越好，False表示值越小越好
    返回: 30-100 的分数
    """
    if not higher_is_better:
        # 反转：值越小越好 → 内部转为越大越好
        value, best, good, bad = -value, -best, -good, -bad

    if value >= best:
        return 100.0
    elif value >= good:
        # best~good 之间线性插值 100~80
        ratio = (value - good) / (best - good) if best != good else 1.0
        return 80.0 + ratio * 20.0
    elif value >= bad:
        # good~bad 之间线性插值 80~30
        ratio = (value - bad) / (good - bad) if good != bad else 1.0
        return 30.0 + ratio * 50.0
    else:
        return 30.0


class ActionScorer:
    """
    动作规范性评分器
    评分维度：角度达标(50%) + 动作幅度(30%) + 过程稳定性(20%)
    """

    def __init__(self):
        self.scores_history = []  # 每次动作的评分历史

    def reset(self):
        self.scores_history = []

    def score_action(self, action_key, rep_angles, current_angles=None):
        """
        对一次完成的动作进行评分
        action_key: 动作类型
        rep_angles: 该次动作记录的关键角度（来自ActionCounter）
        current_angles: 当前帧角度（可选）
        返回: (总分0-100, 提示列表)
        """
        if action_key == "squat":
            return self._score_squat(rep_angles)
        elif action_key == "push-up":
            return self._score_pushup(rep_angles)
        elif action_key == "crunches":
            return self._score_crunches(rep_angles)
        elif action_key == "lunge":
            return self._score_lunge(rep_angles)
        return 0, ["未知动作类型"]

    def get_realtime_tips(self, action_key, angles):
        """
        根据当前帧角度给出实时提示（不等动作完成）
        每种动作都必须有对应分支，否则提示无法被刷新
        """
        tips = []
        if action_key == "squat":
            torso = angles.get("torso_angle", 0)
            if torso > SQUAT_SCORE_RULES["max_torso_lean"] + 10:
                tips.append("注意：背部前倾过大")
        elif action_key == "push-up":
            body = angles.get("body_angle", 180)
            if abs(180 - body) > PUSHUP_SCORE_RULES["max_body_bend"] + 5:
                tips.append("注意：核心不稳，保持身体平直")
        elif action_key == "crunches":
            torso = angles.get("torso_angle", 180)
            curl_amount = 180 - torso
            if curl_amount < 20:
                tips.append("注意：卷腹幅度不足，再用力卷起")
        elif action_key == "lunge":
            torso = angles.get("torso_lean", 0)
            if torso > LUNGE_SCORE_RULES["max_torso_lean"] + 5:
                tips.append("注意：身体重心不稳定")
        return tips

    def _score_squat(self, rep_angles):
        """深蹲评分"""
        tips = []

        min_knee = rep_angles.get("min_knee", 180)

        # 角度达标 (S_angle) — 膝角越小越好
        # ≤80°满分, ≤100°良好, ≤130°及格
        s_angle = _linear_score(min_knee, best=80, good=100, bad=130, higher_is_better=False)
        if min_knee > 120:
            tips.append("下蹲深度不足")
        elif min_knee > 130:
            tips.append("下蹲深度严重不足")

        # 动作幅度 (S_range) — 膝角变化越大越好
        range_angle = 180 - min_knee
        s_range = _linear_score(range_angle, best=90, good=70, bad=40, higher_is_better=True)
        if range_angle < 40:
            tips.append("动作幅度不足")

        # 稳定性 (S_stable) — 躯干倾斜越小越好
        max_torso = rep_angles.get("max_torso_lean", 0)
        s_stable = _linear_score(max_torso, best=15, good=30, bad=50, higher_is_better=False)
        if max_torso > SQUAT_SCORE_RULES["max_torso_lean"]:
            tips.append("背部前倾过大")

        total = (SCORE_WEIGHTS["angle"] * s_angle +
                 SCORE_WEIGHTS["range"] * s_range +
                 SCORE_WEIGHTS["stable"] * s_stable)

        self.scores_history.append(total)
        return total, tips

    def _score_pushup(self, rep_angles):
        """俯卧撑评分"""
        tips = []

        min_elbow = rep_angles.get("min_elbow", 180)

        # 角度达标 — 肘角越小越好
        s_angle = _linear_score(min_elbow, best=70, good=95, bad=120, higher_is_better=False)
        if min_elbow > PUSHUP_SCORE_RULES["min_elbow_angle"]:
            tips.append("下压不充分")

        # 动作幅度
        range_angle = 180 - min_elbow
        s_range = _linear_score(range_angle, best=100, good=75, bad=50, higher_is_better=True)
        if range_angle < 50:
            tips.append("动作幅度不足")

        # 稳定性 — 身体弯曲越小越好
        body_bend = rep_angles.get("body_bend", 0)
        s_stable = _linear_score(body_bend, best=5, good=15, bad=30, higher_is_better=False)
        if body_bend > PUSHUP_SCORE_RULES["max_body_bend"]:
            tips.append("核心不稳，注意收腹")

        total = (SCORE_WEIGHTS["angle"] * s_angle +
                 SCORE_WEIGHTS["range"] * s_range +
                 SCORE_WEIGHTS["stable"] * s_stable)

        self.scores_history.append(total)
        return total, tips

    def _score_crunches(self, rep_angles):
        """卷腹评分"""
        tips = []

        min_torso = rep_angles.get("min_torso", 180)
        curl_amount = 180 - min_torso

        # 角度达标 — 卷腹幅度越大越好
        s_angle = _linear_score(curl_amount, best=65, good=45, bad=20, higher_is_better=True)
        if curl_amount < CRUNCH_SCORE_RULES["min_torso_curl"]:
            tips.append("卷腹幅度不足")

        # 动作幅度
        s_range = _linear_score(curl_amount, best=55, good=35, bad=15, higher_is_better=True)

        # 稳定性 — 与目标角度(100°)的偏差越小越好
        target_min_torso = 100
        torso_deviation = abs(min_torso - target_min_torso)
        s_stable = _linear_score(torso_deviation, best=5, good=20, bad=45, higher_is_better=False)
        if torso_deviation > 40:
            tips.append("动作控制力不足，注意匀速发力")

        total = (SCORE_WEIGHTS["angle"] * s_angle +
                 SCORE_WEIGHTS["range"] * s_range +
                 SCORE_WEIGHTS["stable"] * s_stable)

        self.scores_history.append(total)
        return total, tips

    def _score_lunge(self, rep_angles):
        """弓步蹲评分"""
        tips = []

        min_front_knee = rep_angles.get("min_front_knee", 180)
        max_torso_lean = rep_angles.get("max_torso_lean", 0)

        # 角度达标 — 前腿膝角在 85-95° 最佳
        ideal_knee = 90
        knee_deviation = abs(min_front_knee - ideal_knee)
        s_angle = _linear_score(knee_deviation, best=5, good=15, bad=35, higher_is_better=False)
        if min_front_knee < LUNGE_SCORE_RULES["min_front_knee"]:
            tips.append("前腿膝盖弯曲过深")
        elif min_front_knee > LUNGE_SCORE_RULES["max_front_knee"]:
            tips.append("前腿角度不合适，下蹲不够")

        # 动作幅度
        range_angle = 180 - min_front_knee
        s_range = _linear_score(range_angle, best=90, good=70, bad=40, higher_is_better=True)

        # 稳定性 — 躯干偏斜越小越好
        s_stable = _linear_score(max_torso_lean, best=5, good=15, bad=30, higher_is_better=False)
        if max_torso_lean > LUNGE_SCORE_RULES["max_torso_lean"]:
            tips.append("身体重心不稳定")

        total = (SCORE_WEIGHTS["angle"] * s_angle +
                 SCORE_WEIGHTS["range"] * s_range +
                 SCORE_WEIGHTS["stable"] * s_stable)

        self.scores_history.append(total)
        return total, tips

    def get_average_score(self):
        """获取平均评分"""
        if not self.scores_history:
            return 0.0
        return np.mean(self.scores_history)
