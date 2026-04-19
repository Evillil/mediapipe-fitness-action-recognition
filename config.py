# -*- coding: utf-8 -*-
"""
全局配置文件
基于 MediaPipe 与轻量级深度学习模型的健身动作智能识别系统
"""
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_MATERIAL_DIR = os.path.join(BASE_DIR, "material")
LEGACY_MATERIAL_DIR = os.path.join(os.path.dirname(BASE_DIR), "material")
MATERIAL_DIR = (
    os.getenv("FITNESS_MATERIAL_DIR")
    or (REPO_MATERIAL_DIR if os.path.isdir(REPO_MATERIAL_DIR) else LEGACY_MATERIAL_DIR)
)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "training_records.db")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 确保目录存在
for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 动作类别配置 ====================
ACTION_CLASSES = ["squat", "push-up", "crunches", "lunge"]
ACTION_NAMES_CN = {
    "squat": "深蹲",
    "push-up": "俯卧撑",
    "crunches": "卷腹",
    "lunge": "弓步蹲"
}
NUM_CLASSES = len(ACTION_CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ACTION_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(ACTION_CLASSES)}

# ==================== MediaPipe 关键点配置 ====================
# 从 33 个关键点中选取 17 个核心关键点
# 索引对照 MediaPipe Pose Landmark
SELECTED_LANDMARKS = [
    0,   # 鼻尖 (NOSE)
    11,  # 左肩 (LEFT_SHOULDER)
    12,  # 右肩 (RIGHT_SHOULDER)
    13,  # 左肘 (LEFT_ELBOW)
    14,  # 右肘 (RIGHT_ELBOW)
    15,  # 左腕 (LEFT_WRIST)
    16,  # 右腕 (RIGHT_WRIST)
    23,  # 左髋 (LEFT_HIP)
    24,  # 右髋 (RIGHT_HIP)
    25,  # 左膝 (LEFT_KNEE)
    26,  # 右膝 (RIGHT_KNEE)
    27,  # 左踝 (LEFT_ANKLE)
    28,  # 右踝 (RIGHT_ANKLE)
    29,  # 左脚跟 (LEFT_HEEL)
    30,  # 右脚跟 (RIGHT_HEEL)
    31,  # 左脚尖 (LEFT_FOOT_INDEX)
    32,  # 右脚尖 (RIGHT_FOOT_INDEX)
]
NUM_LANDMARKS = len(SELECTED_LANDMARKS)  # 17
COORD_DIM = 3  # x, y, z
NUM_ANGLE_FEATURES = 6  # 左右肘角、左右膝角、髋部夹角、躯干倾角
FEATURE_DIM = NUM_LANDMARKS * COORD_DIM + NUM_ANGLE_FEATURES  # 51 + 6 = 57

# 关键点在17点子集中的索引映射
# 用于角度计算
KP_NOSE = 0
KP_LEFT_SHOULDER = 1
KP_RIGHT_SHOULDER = 2
KP_LEFT_ELBOW = 3
KP_RIGHT_ELBOW = 4
KP_LEFT_WRIST = 5
KP_RIGHT_WRIST = 6
KP_LEFT_HIP = 7
KP_RIGHT_HIP = 8
KP_LEFT_KNEE = 9
KP_RIGHT_KNEE = 10
KP_LEFT_ANKLE = 11
KP_RIGHT_ANKLE = 12
KP_LEFT_HEEL = 13
KP_RIGHT_HEEL = 14
KP_LEFT_FOOT = 15
KP_RIGHT_FOOT = 16

# ==================== 滑动窗口参数 ====================
WINDOW_SIZE = 30   # 窗口长度（帧数）
WINDOW_STEP = 10   # 滑动步长（帧数）

# 训练集每类最大样本数（含增强），防止某类视频过多导致类别不平衡加剧过拟合
# 设为 0 表示不限制
MAX_TRAIN_SAMPLES_PER_CLASS = 18500

# ==================== 预处理参数 ====================
SMOOTH_ALPHA = 0.6       # 指数滑动平均系数
VISIBILITY_THRESHOLD = 0.5  # 可见度阈值

# ==================== 视频采集参数 ====================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

# ==================== 模型训练参数 ====================
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 80
DROPOUT_RATE = 0.3
LSTM_HIDDEN_SIZE = 64
CNN_CHANNELS_1 = 64
CNN_CHANNELS_2 = 128
CNN_KERNEL_SIZE = 3
LR_PATIENCE = 5      # 学习率衰减等待轮数
LR_FACTOR = 0.5       # 学习率衰减因子
EARLY_STOP_PATIENCE = 15  # 早停等待轮数

# ==================== 推理参数 ====================
PREDICTION_SMOOTH_WINDOW = 5   # 预测结果平滑窗口
CONFIDENCE_THRESHOLD = 0.6     # 置信度阈值

# ==================== MediaPipe 检测参数 ====================
POSE_DETECTION_CONFIDENCE = 0.5    # 姿态检测置信度阈值
POSE_PRESENCE_CONFIDENCE = 0.5     # 姿态存在置信度阈值
POSE_TRACKING_CONFIDENCE = 0.5     # 姿态追踪置信度阈值（与 MediaPipe 默认值一致，优先稳定性）
SKELETON_FALLBACK_FRAMES = 5       # 骨骼检测失败时，最多沿用前N帧数据

# ==================== 状态机参数（动作计数）====================
STATE_CONFIRM_FRAMES = 3  # 状态确认帧数

# 深蹲参数
SQUAT_KNEE_ANGLE_UP = 140      # 站立膝角阈值（原150过严，快速深蹲站起不完全伸直）
SQUAT_KNEE_ANGLE_DOWN = 120    # 下蹲膝角阈值（原100过严，很多标准深蹲膝角在100-120之间）
SQUAT_MIN_ACTIVE_FRAMES = 5    # 深蹲最小 ACTIVE 持续帧数（过滤瞬间角度抖动假触发）
SQUAT_MIN_ROM = 15             # 深蹲最小动作幅度（站起膝角 - 最低膝角 需超过此值）

# 俯卧撑参数
PUSHUP_ELBOW_ANGLE_UP = 150    # 撑起肘角阈值
PUSHUP_ELBOW_ANGLE_DOWN = 100  # 下压肘角阈值
PUSHUP_MIN_ROM = 30            # 单次俯卧撑最小动作幅度（肘角变化需超过此值才计为有效）
PUSHUP_MIN_ACTIVE_FRAMES = 8   # 俯卧撑最小 ACTIVE 持续帧数（过滤 MediaPipe 单帧跳变假计数）

# 卷腹参数
# 实测：不同视频角度下平躺躯干角范围约 120-170°，卷起约 85-110°
# 阈值需兼容不同拍摄角度和体型
CRUNCH_TORSO_ANGLE_REST = 120  # 平躺躯干角阈值（回到此角度以上即视为平躺）
CRUNCH_TORSO_ANGLE_UP = 110    # 卷起躯干角阈值（低于此角度即视为卷起）

# 弓步蹲参数
LUNGE_KNEE_ANGLE_UP = 140      # 站立膝角阈值（原150过严）
LUNGE_KNEE_ANGLE_DOWN = 110    # 弓步膝角阈值（原100过严）

# ==================== 评分参数 ====================
SCORE_WEIGHTS = {
    "angle": 0.5,
    "range": 0.3,
    "stable": 0.2
}

# 深蹲评分阈值
SQUAT_SCORE_RULES = {
    "min_knee_angle": 110,      # 膝角>110 => 下蹲深度不足
    "max_torso_lean": 35,       # 躯干前倾>35度 => 背部前倾过大
}

# 俯卧撑评分阈值
PUSHUP_SCORE_RULES = {
    "min_elbow_angle": 95,      # 肘角>95 => 下压不充分
    "max_body_bend": 15,        # 身体弯曲>15度 => 核心不稳
}

# 卷腹评分阈值
CRUNCH_SCORE_RULES = {
    "min_torso_curl": 30,       # 卷腹幅度不足
    "max_speed_var": 0.5,       # 速度变化过大 => 动作过快
}

# 弓步蹲评分阈值
LUNGE_SCORE_RULES = {
    "min_front_knee": 80,       # 膝角范围下限
    "max_front_knee": 100,      # 膝角范围上限
    "max_torso_lean": 15,       # 躯干偏斜阈值
}
