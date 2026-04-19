# -*- coding: utf-8 -*-
"""
健身动作智能识别系统 - Streamlit 主界面
基于 MediaPipe 与轻量级深度学习模型
适配 MediaPipe 0.10+ Tasks API
"""
import os
import sys
import time
import html
import tempfile
from collections import deque
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import streamlit as st
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from utils import draw_skeleton, put_chinese_text
from inference import ActionInference
from action_counter import ActionCounter
from action_scorer import ActionScorer
from database import Database
from ai_store import AIStore
from provider_ai_client import (
    AIConfigError,
    AIResponseError,
    MultiProviderAIClient,
    PROVIDER_OPTIONS,
    get_provider_option,
    provider_default_base_url,
    provider_label,
)
from report_service import (
    AICoachError,
    AIReportError,
    build_report_pdf_bytes,
    generate_ai_coach_reply,
    generate_ai_training_report,
)

# ==================== 摄像头检测 ====================
def detect_cameras(max_index=10):
    """检测可用的摄像头设备"""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            name = f"摄像头 {i} ({w}x{h})"
            available.append({"index": i, "name": name, "width": w, "height": h})
            cap.release()
    return available


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="健身动作智能识别系统",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS ====================
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        padding: 0.5rem;
    }
    .sub-title { text-align: censtter; color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 1.2rem; text-align: center;
        border: 1px solid #2a2a4a; margin: 0.3rem 0;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 0.3rem; }
    .tip-box {
        background: linear-gradient(135deg, #ff6b6b20 0%, #ee585830 100%);
        border-left: 4px solid #ff6b6b; border-radius: 8px;
        padding: 0.8rem 1rem; margin: 0.5rem 0; color: #ff6b6b; font-size: 0.95rem;
    }
    .good-box {
        background: linear-gradient(135deg, #51cf6620 0%, #2ecc7130 100%);
        border-left: 4px solid #51cf66; border-radius: 8px;
        padding: 0.8rem 1rem; margin: 0.5rem 0; color: #51cf66; font-size: 0.95rem;
    }
    .status-active {
        display: inline-block; width: 10px; height: 10px;
        background: #00ff88; border-radius: 50%; margin-right: 6px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: visible;}
</style>
""", unsafe_allow_html=True)


# ==================== 初始化 ====================
@st.cache_resource
def init_pose_landmarker_image():
    """初始化 MediaPipe PoseLandmarker (IMAGE模式，用于上传视频逐帧处理)"""
    model_path = os.path.join(MODEL_DIR, "pose_landmarker_heavy.task")
    if not os.path.exists(model_path):
        st.error(f"❌ 姿态模型文件不存在: {model_path}")
        return None
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=POSE_PRESENCE_CONFIDENCE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def create_pose_landmarker_video():
    """
    创建 MediaPipe PoseLandmarker (VIDEO模式，用于摄像头连续流，带时序追踪)
    注意：不使用 @st.cache_resource，每次新会话需要全新实例以重置内部追踪状态和时间戳
    """
    model_path = os.path.join(MODEL_DIR, "pose_landmarker_heavy.task")
    if not os.path.exists(model_path):
        st.error(f"❌ 姿态模型文件不存在: {model_path}")
        return None
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
    )
    return vision.PoseLandmarker.create_from_options(options)


@st.cache_resource
def init_inference():
    """初始化推理引擎"""
    model_path = os.path.join(MODEL_DIR, "best_cnn_lstm.pth")
    return ActionInference(model_path=model_path)


@st.cache_resource
def init_database(_version=12):
    """初始化数据库（修改 _version 强制刷新缓存）"""
    return Database()


@st.cache_resource
def init_ai_store(_version=3):
    """初始化 AI 存储"""
    return AIStore()


def init_session_state():
    """初始化会话状态"""
    defaults = {
        "is_running": False,
        "counter": ActionCounter(),
        "scorer": ActionScorer(),
        "selected_action": "squat",
        "current_score": 0,
        "current_tips": [],
        "last_score": 0,
        "start_time": None,
        "fps_list": deque(maxlen=30),
        "frame_count": 0,
        "pending_save": None,
        "logged_in": False,
        "username": "",
        "user_role": "",
        "user_id": None,
        "selected_ai_report_id": None,
        "selected_ai_chat_report_id": None,
        "selected_ai_chat_session_id": None,
        "pending_delete_ai_report_id": None,
        "pending_delete_ai_chat_session_id": None,
        "pending_select_ai_chat_session_id": None,
        "pending_delete_record_ids": [],
        "history_last_select_all": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_current_user_id():
    """获取当前登录用户 ID"""
    try:
        value = int(st.session_state.get("user_id"))
        return value if value > 0 else None
    except Exception:
        return None


def get_current_username(default="user"):
    """获取当前登录用户名"""
    username = st.session_state.get("username", default)
    if username is None:
        return default
    username = str(username).strip()
    return username or default


def mask_secret(secret):
    """掩码显示密钥"""
    if not secret:
        return "未设置"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * max(len(secret) - 8, 4)}{secret[-4:]}"


def render_report_section(title, items, empty_text="暂无内容"):
    """渲染报告分节"""
    st.markdown(f"#### {title}")
    if not items:
        st.info(empty_text)
        return
    for item in items:
        st.markdown(f"- {item}")


def render_ai_report_content(report_detail):
    """渲染 AI 报告详情"""
    report = report_detail.get("report_json", {}) if report_detail else {}
    source = report_detail.get("source_summary", {}) if report_detail else {}
    training_summary = source.get("training_summary", {})
    plan_summary = source.get("plan_summary", {})
    profile = source.get("profile", {})

    st.markdown(f"### {report.get('report_title', 'AI 训练分析报告')}")
    st.caption(
        f"生成时间：{report_detail.get('report_time', '')} | "
        f"模型：{report_detail.get('provider_name', '')} / {report_detail.get('model_name', '')}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("分析记录数", f"{training_summary.get('recent_record_count', 0)} 条")
    c2.metric("累计训练次数", f"{training_summary.get('total_sessions', 0)} 次")
    c3.metric("累计动作数", f"{training_summary.get('total_reps', 0)} 个")
    c4.metric("计划完成率", f"{plan_summary.get('completion_rate', 0):.1f}%")

    summary = report.get("summary", "").strip()
    if summary:
        st.markdown(f'<div class="good-box"><b>总结：</b>{summary}</div>', unsafe_allow_html=True)

    if profile:
        st.caption(
            f"目标：{profile.get('fitness_goal', '未填写')} | "
            f"BMI：{profile.get('bmi', 0)} ({profile.get('bmi_status', '未知')})"
        )

    render_report_section("用户概况", report.get("user_profile", []), "未生成用户概况分析。")
    render_report_section("训练总览", report.get("training_overview", []), "未生成训练总览。")
    render_report_section("目标匹配度分析", report.get("goal_alignment", []), "未生成目标匹配度分析。")
    render_report_section("计划执行情况", report.get("plan_execution", []), "当前没有计划执行分析。")
    render_report_section("主要问题", report.get("key_problems", []), "当前没有明确识别出主要问题。")
    render_report_section("训练建议", report.get("training_suggestions", []), "未生成训练建议。")
    render_report_section("未来 7 天行动清单", report.get("next_7_days_actions", []), "未生成行动清单。")
    render_report_section("风险与注意事项", report.get("risk_alerts", []), "当前没有额外风险提示。")


def build_ai_report_pdf_filename(report_detail):
    """构建 PDF 导出文件名"""
    report = report_detail.get("report_json", {}) if report_detail else {}
    title = str(report.get("report_title") or "AI专属私教训练报告").strip()
    safe_title = "".join(ch if ch.isalnum() or ch in ("_", "-", " ") else "_" for ch in title)
    safe_title = "_".join(safe_title.split()) or "AI专属私教训练报告"
    report_time = str(report_detail.get("report_time") or "").replace(":", "-").replace(" ", "_")
    return f"{safe_title}_{report_time or 'report'}.pdf"


def format_ai_chat_session_label(session_item):
    """格式化聊天会话标签"""
    title = str(session_item.get("session_title") or "新聊天").strip() or "新聊天"
    updated_at = str(session_item.get("updated_at") or "").strip()
    message_count = int(session_item.get("message_count") or 0)
    short_time = updated_at[5:16] if len(updated_at) >= 16 else updated_at
    return f"{title} · {short_time} · {message_count}条"


def _format_chat_html(content):
    text = html.escape(str(content or "").strip())
    return text.replace("\n", "<br>") if text else "&nbsp;"


def render_ai_chat_messages(messages):
    """渲染 AI 问答记录"""
    if not messages:
        st.info("当前报告还没有问答记录。可以直接向 AI 私教追问。")
        return

    rows = []
    for item in messages:
        role = "user" if item.get("role") == "user" else "assistant"
        role_label = "你" if role == "user" else "AI 私教"
        created_at = str(item.get("created_at") or "").strip()
        meta = f"{role_label} · {created_at}" if created_at else role_label
        rows.append(
            (
                f'<div class="coach-chat-row {role}">'
                f'<div class="coach-chat-card">'
                f'<div class="coach-chat-meta">{meta}</div>'
                f'<div class="coach-chat-bubble {role}">{_format_chat_html(item.get("content", ""))}</div>'
                f'</div></div>'
            )
        )

    chat_html = """
    <style>
    .coach-chat-scroll {
        padding: 0.25rem 0.2rem 0.75rem 0.2rem;
    }
    .coach-chat-row {
        display: flex;
        margin: 0 0 0.95rem 0;
    }
    .coach-chat-row.user {
        justify-content: flex-end;
    }
    .coach-chat-row.assistant {
        justify-content: flex-start;
    }
    .coach-chat-card {
        max-width: 78%;
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
    }
    .coach-chat-row.user .coach-chat-card {
        align-items: flex-end;
    }
    .coach-chat-row.assistant .coach-chat-card {
        align-items: flex-start;
    }
    .coach-chat-meta {
        font-size: 0.76rem;
        color: #7b8794;
        padding: 0 0.35rem;
    }
    .coach-chat-bubble {
        border-radius: 18px;
        padding: 0.9rem 1rem;
        line-height: 1.7;
        font-size: 0.98rem;
        word-break: break-word;
        white-space: normal;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }
    .coach-chat-bubble.user {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
        color: #ffffff;
        border-bottom-right-radius: 6px;
    }
    .coach-chat-bubble.assistant {
        background: #f8fafc;
        color: #24324a;
        border: 1px solid #e2e8f0;
        border-bottom-left-radius: 6px;
    }
    </style>
    """ + f'<div class="coach-chat-scroll">{"".join(rows)}</div>'

    with st.container(height=520, border=True):
        st.html(chat_html)


# ==================== 主界面 ====================
def render_main_page():
    """渲染主训练页面"""
    st.markdown('<h1 class="main-title">🏋️ 健身动作智能识别系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">基于 MediaPipe 与轻量级深度学习模型 | 本地化运行 · 隐私安全</p>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ 系统设置")
        st.divider()

        source_type = st.radio("📹 视频输入源", ["摄像头", "上传视频"], index=0)

        camera_index = 0
        uploaded_file = None

        cam_resolution = (640, 480)
        if source_type == "摄像头":
            # 检测可用摄像头
            if "cameras" not in st.session_state or not st.session_state.cameras:
                st.session_state.cameras = detect_cameras()

            if st.button("🔄 刷新摄像头列表", use_container_width=True):
                st.session_state.cameras = detect_cameras()
                st.rerun()

            cameras = st.session_state.cameras
            if cameras:
                cam_options = {cam["name"]: cam["index"] for cam in cameras}
                selected_cam = st.selectbox(
                    "选择摄像头",
                    list(cam_options.keys()),
                    help="系统自动检测到的可用摄像头"
                )
                camera_index = cam_options[selected_cam]
            else:
                st.warning("⚠️ 未检测到摄像头，请连接后点击刷新")
                camera_index = st.number_input("手动输入摄像头编号", min_value=0, max_value=10, value=0)

            # 摄像头分辨率选择
            resolution_options = {
                "1280×720 (HD 横屏)": (1280, 720),
                "1920×1080 (FHD 横屏)": (1920, 1080),
                "720×1280 (HD 竖屏)": (720, 1280),
                "1080×1920 (FHD 竖屏)": (1080, 1920),
                "640×480 (VGA)": (640, 480),
                "自定义": None,
            }
            selected_res = st.selectbox("📐 摄像头分辨率", list(resolution_options.keys()), index=0,
                                        help="横屏：电脑摄像头/横放手机 | 竖屏：竖放手机/OBS竖屏源")
            if resolution_options[selected_res] is not None:
                cam_resolution = resolution_options[selected_res]
            else:
                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    custom_w = st.number_input("宽度", min_value=320, max_value=3840, value=1280, step=10)
                with res_c2:
                    custom_h = st.number_input("高度", min_value=240, max_value=2160, value=720, step=10)
                cam_resolution = (custom_w, custom_h)
            if cam_resolution[0] < cam_resolution[1]:
                st.caption("📱 竖屏模式 — 适合手机竖放拍摄")
            else:
                st.caption("🖥️ 横屏模式 — 适合电脑摄像头/横放设备")
        else:
            uploaded_file = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov", "mkv"])

        st.divider()
        st.markdown("### 🎯 动作选择")
        action_options = {
            "深蹲 (Squat)": "squat",
            "俯卧撑 (Push-up)": "push-up",
            "卷腹 (Crunches)": "crunches",
            "弓步蹲 (Lunge)": "lunge"
        }
        selected = st.selectbox("选择训练动作", list(action_options.keys()))
        st.session_state.selected_action = action_options[selected]

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("▶️ 开始", use_container_width=True, type="primary")
        with col2:
            stop_btn = st.button("⏹️ 停止", use_container_width=True)

        if start_btn:
            st.session_state.is_running = True
            st.session_state.counter = ActionCounter()
            st.session_state.counter.reset(st.session_state.selected_action)
            st.session_state.scorer = ActionScorer()
            st.session_state.start_time = time.time()
            st.session_state.fps_list = deque(maxlen=30)
            st.session_state.frame_count = 0
            st.session_state.current_tips = []
            st.session_state.current_score = 0
            st.session_state.last_score = 0

        if stop_btn and st.session_state.is_running:
            st.session_state.is_running = False
            # 将训练数据暂存到 session_state，等待用户确认保存
            counter = st.session_state.counter
            scorer = st.session_state.scorer
            if counter.count > 0:
                duration = time.time() - st.session_state.start_time if st.session_state.start_time else 0
                action_type = st.session_state.selected_action
                action_cn = ACTION_NAMES_CN.get(action_type, action_type)
                avg_score = scorer.get_average_score()
                st.session_state.pending_save = {
                    "action_cn": action_cn,
                    "count": counter.count,
                    "avg_score": avg_score,
                    "duration": duration,
                }

        st.divider()
        with st.expander("📖 使用说明"):
            st.markdown("""
            1. 选择**视频输入源**（摄像头/上传视频）
            2. 选择要训练的**动作类型**  
            3. 点击 **开始** 按钮
            4. 面对摄像头，使**全身**进入画面
            5. 系统将对所选动作进行计数、评分和给出纠错建议
            6. 训练结束后点击 **停止**
            """)

    # 显示待保存的训练记录对话框
    if st.session_state.pending_save:
        data = st.session_state.pending_save
        st.success(f"🏁 训练完成！{data['action_cn']} {data['count']}次，平均评分 {data['avg_score']:.1f}")
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            if st.button("💾 保存训练记录", use_container_width=True, type="primary"):
                db = init_database()
                db.save_training_record(
                    action_type=data["action_cn"], repetitions=data["count"],
                    avg_score=data["avg_score"], duration_sec=data["duration"],
                    username=get_current_username(),
                    user_id=get_current_user_id(),
                )
                st.session_state.pending_save = None
                st.toast("✅ 训练记录已保存！")
                st.rerun()
        with save_col2:
            if st.button("🗑️ 不保存", use_container_width=True):
                st.session_state.pending_save = None
                st.rerun()
        return

    if source_type == "摄像头":
        process_video_source(camera_index=camera_index, cam_resolution=cam_resolution)
    else:
        process_video_source(uploaded_file=uploaded_file)


def process_video_source(camera_index=None, uploaded_file=None, cam_resolution=(640, 480)):
    """统一的视频处理流程"""
    # 指标区域
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    metric_action = col_m1.empty()
    metric_count = col_m2.empty()
    metric_score = col_m3.empty()
    metric_fps = col_m4.empty()

    col_video, col_info = st.columns([3, 1])
    with col_video:
        video_placeholder = st.empty()
        progress_bar = None
        if uploaded_file:
            progress_bar = st.progress(0)
    with col_info:
        tips_placeholder = st.empty()
        status_placeholder = st.empty()
        angles_placeholder = st.empty()

    if not st.session_state.is_running:
        video_placeholder.info("👈 点击左侧「开始」按钮启动识别")
        _update_metrics(metric_action, metric_count, metric_score, metric_fps, "等待中", 0, 0, 0)
        return

    # 初始化组件（根据视频源选择不同模式）
    is_camera = (uploaded_file is None)
    if is_camera:
        # VIDEO模式：每次新建实例以重置内部追踪状态和时间戳序列
        landmarker = create_pose_landmarker_video()
    else:
        landmarker = init_pose_landmarker_image()  # IMAGE模式
    if landmarker is None:
        st.error("姿态估计模型未初始化")
        return

    inference_engine = init_inference()
    inference_engine.reset()
    counter = st.session_state.counter
    scorer = st.session_state.scorer

    # 打开视频源
    cap = None
    temp_path = None

    if uploaded_file is not None:
        os.makedirs(LOG_DIR, exist_ok=True)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=LOG_DIR)
        tfile.write(uploaded_file.read())
        tfile.close()
        temp_path = tfile.name
        cap = cv2.VideoCapture(temp_path)
        source_label = "上传视频"
    else:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_resolution[1])
        source_label = f"摄像头 {camera_index} ({cam_resolution[0]}x{cam_resolution[1]})"

    if not cap.isOpened():
        st.error(f"❌ 无法打开{source_label}")
        st.session_state.is_running = False
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if uploaded_file else 0
    status_placeholder.markdown(f'<div class="good-box">🟢 {source_label}已连接 - 识别运行中</div>',
                                unsafe_allow_html=True)

    frame_idx = 0
    last_result = None
    last_action_class = None
    last_raw_pixel = None
    last_vis = None
    last_raw_3d = None
    missed_frames = 0        # 连续检测失败帧数
    MAX_PROCESS_WIDTH = 640
    video_start_time = time.monotonic()  # VIDEO模式时间戳基准
    last_ts_ms = -1  # 上一帧时间戳，确保严格递增

    # 上传视频模式：隔帧检测MediaPipe + 线性插值坐标
    # 摄像头模式：每帧检测（实时性要求高）
    SKIP_INTERVAL = 1 if is_camera else 2  # 每N帧检测一次MediaPipe

    try:
        while st.session_state.is_running:
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                if uploaded_file:
                    st.session_state.is_running = False
                    status_placeholder.markdown('<div class="good-box">✅ 视频分析完成！</div>',
                                                unsafe_allow_html=True)
                else:
                    st.warning("⚠️ 摄像头读取失败")
                break

            frame_idx += 1
            if progress_bar and total_frames > 0 and frame_idx % 10 == 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

            # === 智能隔帧：上传视频每2帧检测1次MediaPipe，非检测帧用线性插值 ===
            do_detect = (frame_idx % SKIP_INTERVAL == 1) or (SKIP_INTERVAL == 1) or last_result is None
            # UI 刷新频率控制（上传视频每8帧刷一次，提前计算供骨骼绘制判断）
            do_update = is_camera or (frame_idx % 8 == 0) or (total_frames > 0 and frame_idx >= total_frames)

            # 仅在检测帧才做 resize / flip（非检测帧跳过，节省开销）
            process_frame = None
            if do_detect:
                h_orig, w_orig = frame.shape[:2]
                if w_orig > MAX_PROCESS_WIDTH:
                    scale = MAX_PROCESS_WIDTH / w_orig
                    process_frame = cv2.resize(frame, (MAX_PROCESS_WIDTH, int(h_orig * scale)))
                else:
                    process_frame = frame
                # 摄像头模式：先对处理帧进行镜像翻转再送入 MediaPipe
                if is_camera:
                    process_frame = cv2.flip(process_frame, 1)

            result = None
            if do_detect:
                rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                try:
                    if is_camera:
                        ts_ms = int((time.monotonic() - video_start_time) * 1000)
                        if ts_ms <= last_ts_ms:
                            ts_ms = last_ts_ms + 1
                        last_ts_ms = ts_ms
                        result = landmarker.detect_for_video(mp_image, ts_ms)
                    else:
                        result = landmarker.detect(mp_image)
                    last_result = result
                except Exception:
                    result = last_result
            else:
                result = last_result

            action_class = last_action_class
            confidence = 0
            current_angles = {}
            raw_pixel = last_raw_pixel
            vis = last_vis
            raw_3d = last_raw_3d

            # 判断本帧是否成功检测到姿态
            detect_ok = (result is not None and result.pose_landmarks
                         and len(result.pose_landmarks) > 0)

            if detect_ok:
                # 检测恢复后清空"请调整位置"等检测失败提示
                if missed_frames > 0:
                    st.session_state.current_tips = []
                missed_frames = 0

                if do_detect:
                    # 检测帧：完整推理，更新所有数据
                    action_class, confidence, raw_pixel, vis, raw_3d = \
                        inference_engine.process_landmarks_from_result(result)

                    last_action_class = action_class
                    last_raw_pixel = raw_pixel
                    last_vis = vis
                    last_raw_3d = raw_3d
                else:
                    # 非检测帧：raw_3d 直接沿用上一帧真实检测值（不做线性外推）
                    # 之前的 last_raw_3d + (last_raw_3d - prev_raw_3d) * 0.5 线性外推
                    # 在动作方向反转点（如深蹲最低点）会严重过冲：
                    #   - 污染 min_knee/min_elbow，导致 ROM 虚高
                    #   - 提前触碰 DOWN/UP 角度阈值，状态机时机失真
                    # counter 内部已有 EMA 平滑（action_counter.py 的 _sanitize_angle），
                    # 对重复输入天然吸收，不需要外推辅助
                    # raw_pixel/vis 也沿用上一帧（仅用于绘制，精度要求低）
                    pass

                # 绘制骨骼（仅在需要显示的帧绘制，上传视频非显示帧跳过绘制）
                if raw_pixel is not None and do_update:
                    if is_camera:
                        frame = cv2.flip(frame, 1)
                    draw_skeleton(frame, raw_pixel, vis)

                # 确定当前动作
                current_action_key = st.session_state.selected_action
                if current_action_key and raw_3d is not None:
                    completed, current_angles = counter.update(raw_3d, current_action_key)

                    if completed and counter.angle_history:
                        rep_angles = counter.angle_history[-1]
                        score, tips = scorer.score_action(current_action_key, rep_angles)
                        st.session_state.current_score = score
                        st.session_state.current_tips = tips
                        st.session_state.last_score = score

                    realtime_tips = scorer.get_realtime_tips(current_action_key, current_angles)
                    if realtime_tips:
                        st.session_state.current_tips = realtime_tips
            else:
                # 检测失败：在允许范围内沿用上一帧骨骼数据
                missed_frames += 1

                if missed_frames <= SKELETON_FALLBACK_FRAMES and last_raw_pixel is not None:
                    if do_update:
                        if is_camera:
                            frame = cv2.flip(frame, 1)
                        faded_vis = last_vis * 0.6 if last_vis is not None else None
                        draw_skeleton(frame, last_raw_pixel, faded_vis)
                else:
                    st.session_state.current_tips = ["请调整位置，使全身进入画面"]

            # FPS
            frame_time = time.time() - frame_start
            fps = 1.0 / max(frame_time, 0.001)
            st.session_state.fps_list.append(fps)
            st.session_state.frame_count += 1

            # 动作名称
            display_action = ACTION_NAMES_CN.get(st.session_state.selected_action, "等待中...")

            # === UI 刷新 ===
            if do_update:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                avg_fps = np.mean(st.session_state.fps_list) if st.session_state.fps_list else 0
                _update_metrics(metric_action, metric_count, metric_score, metric_fps,
                                display_action, counter.count, st.session_state.last_score, avg_fps)
                _update_tips(tips_placeholder, st.session_state.current_tips, st.session_state.last_score)
                if current_angles:
                    _update_angles(angles_placeholder, current_angles)

    except Exception as e:
        st.error(f"运行错误: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        cap.release()
        # VIDEO模式的 landmarker 不缓存，需要手动关闭释放资源
        if is_camera and landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass
        if temp_path:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        # 上传视频模式自动保存（不弹窗）
        if uploaded_file is not None:
            _save_training_record()


def _update_metrics(m_action, m_count, m_score, m_fps, action, count, score, fps):
    """更新指标显示"""
    m_action.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color: #00d4ff; font-size: 1.5rem;">{action}</div>
        <div class="metric-label">当前动作</div></div>""", unsafe_allow_html=True)
    m_count.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color: #ffd700;">{count}</div>
        <div class="metric-label">完成次数</div></div>""", unsafe_allow_html=True)
    score_color = "#51cf66" if score >= 80 else "#ffd43b" if score >= 60 else "#ff6b6b"
    m_score.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color: {score_color};">{score:.0f}</div>
        <div class="metric-label">动作评分</div></div>""", unsafe_allow_html=True)
    m_fps.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color: #aaa; font-size: 1.5rem;">{fps:.1f}</div>
        <div class="metric-label">FPS</div></div>""", unsafe_allow_html=True)


def _update_tips(placeholder, tips, score):
    if not tips:
        if score >= 80:
            placeholder.markdown('<div class="good-box">✅ 动作标准！继续保持</div>', unsafe_allow_html=True)
        else:
            placeholder.markdown('<div class="good-box">🎯 开始训练吧</div>', unsafe_allow_html=True)
    else:
        tips_html = "".join(f'<div class="tip-box">⚠️ {tip}</div>' for tip in tips)
        placeholder.markdown(tips_html, unsafe_allow_html=True)


def _update_angles(placeholder, angles):
    name_map = {"knee_angle": "膝关节角度", "torso_angle": "躯干倾角", "elbow_angle": "肘关节角度",
                "body_angle": "身体直线度", "front_knee_angle": "前腿膝角", "torso_lean": "躯干偏斜"}
    lines = []
    for key, val in angles.items():
        display_name = name_map.get(key, key)
        if isinstance(val, (int, float)):
            lines.append(f"**{display_name}**: {val:.1f}°")
    if lines:
        placeholder.markdown("**📐 关节角度**\n\n" + "\n\n".join(lines))


def _save_training_record():
    counter = st.session_state.counter
    scorer = st.session_state.scorer
    if counter.count > 0:
        duration = time.time() - st.session_state.start_time if st.session_state.start_time else 0
        action_type = st.session_state.selected_action
        if not action_type:
            st.error("未选择训练动作，无法保存训练记录")
            return
        action_cn = ACTION_NAMES_CN.get(action_type, action_type)
        avg_score = scorer.get_average_score()
        db = init_database()
        db.save_training_record(action_type=action_cn, repetitions=counter.count,
                                avg_score=avg_score, duration_sec=duration,
                                username=get_current_username(),
                                user_id=get_current_user_id())
        st.success(f"✅ 训练记录已保存：{action_cn} {counter.count}次，平均评分 {avg_score:.1f}")


# ==================== 历史记录页面 ====================
def render_history_page():
    st.markdown('<h1 class="main-title">📊 训练历史记录</h1>', unsafe_allow_html=True)

    # 上一次 rerun 请求重置"全选"状态 —— 必须在 checkbox 实例化之前清理 key，
    # 否则会触发 "cannot be modified after the widget ... is instantiated"
    if st.session_state.pop("_reset_select_all_records", False):
        st.session_state.pop("select_all_records", None)
        st.session_state["history_last_select_all"] = False

    db = init_database()
    cur_user = get_current_username()
    cur_user_id = get_current_user_id()
    stats = db.get_statistics(username=cur_user, user_id=cur_user_id)

    st.markdown("### 📈 训练总览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总训练次数", f"{stats['total_sessions']} 次")
    c2.metric("总动作次数", f"{stats['total_reps']} 个")
    c3.metric("总训练时长", f"{stats['total_duration'] / 60:.1f} 分钟")
    c4.metric("平均评分", f"{stats['avg_score']:.1f} 分")

    st.divider()
    if stats['per_action']:
        st.markdown("### 🏋️ 各动作统计")
        for action in stats['per_action']:
            ca, cb, cc, cd = st.columns(4)
            ca.write(f"**{action[0]}**")
            cb.write(f"训练 {action[1]} 次")
            cc.write(f"共 {action[2]} 个动作")
            cd.write(f"均分 {action[3]:.1f}")

    st.divider()
    st.markdown("### 📝 详细记录")
    cur_user = get_current_username()
    cur_user_id = get_current_user_id()
    records = db.get_training_records(limit=50, username=cur_user, user_id=cur_user_id)
    if records:
        import pandas as pd
        df = pd.DataFrame(records, columns=["ID", "日期", "动作类型", "次数", "平均分", "时长(秒)", "备注"])
        df["时长(秒)"] = df["时长(秒)"].apply(lambda x: f"{x:.1f}")
        record_ids = df["ID"].tolist()

        # 删除功能
        del_col1, del_col2 = st.columns([2, 1])
        with del_col1:
            select_all = st.checkbox("全选", key="select_all_records")
        with del_col2:
            delete_selected_btn = st.button("🗑️ 删除选中", type="secondary")

        for rid in record_ids:
            checkbox_key = f"rec_{rid}"
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = bool(select_all)

        if st.session_state.get("history_last_select_all") != bool(select_all):
            for rid in record_ids:
                st.session_state[f"rec_{rid}"] = bool(select_all)
            st.session_state["history_last_select_all"] = bool(select_all)

        # 记录选择
        selected_ids = []
        for _, row in df.iterrows():
            rid = row["ID"]
            label = f'{row["日期"]}　{row["动作类型"]}　{row["次数"]}次　{row["平均分"]}分'
            checked = st.checkbox(label, key=f"rec_{rid}")
            if checked:
                selected_ids.append(rid)

        # 删除选中
        if delete_selected_btn and selected_ids:
            st.session_state["pending_delete_record_ids"] = list(selected_ids)
        elif delete_selected_btn and not selected_ids:
            st.warning("请先勾选要删除的记录")

        pending_delete_ids = [
            rid for rid in st.session_state.get("pending_delete_record_ids", [])
            if rid in record_ids
        ]
        if pending_delete_ids:
            st.warning(f"确认删除当前选中的 {len(pending_delete_ids)} 条训练记录？此操作不可撤销。")
            cf1, cf2 = st.columns(2)
            with cf1:
                if st.button("确认删除选中", type="primary", use_container_width=True):
                    for rid in pending_delete_ids:
                        db.delete_record(rid)
                        st.session_state.pop(f"rec_{rid}", None)
                    st.session_state["pending_delete_record_ids"] = []
                    # 标记在下一次 rerun 开头重置"全选"相关状态
                    # （不能在这里直接赋值，因为当前 run 里 checkbox 已经实例化过）
                    st.session_state["_reset_select_all_records"] = True
                    st.toast(f"✅ 已删除 {len(pending_delete_ids)} 条记录")
                    st.rerun()
            with cf2:
                if st.button("取消删除", use_container_width=True):
                    st.session_state["pending_delete_record_ids"] = []
                    st.rerun()
    else:
        st.session_state["pending_delete_record_ids"] = []
        st.session_state.pop("select_all_records", None)
        st.session_state["history_last_select_all"] = False
        st.info("暂无训练记录，开始训练后数据将自动保存")


# ==================== 数据可视化页面 ====================
def render_visualization_page():
    """渲染数据可视化页面"""
    st.markdown('<h1 class="main-title">📈 训练数据可视化</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">训练趋势分析 · 多维度数据洞察</p>', unsafe_allow_html=True)

    db = init_database()
    cur_user = get_current_username()
    cur_user_id = get_current_user_id()
    records = db.get_training_records(limit=500, username=cur_user, user_id=cur_user_id)

    if not records:
        st.info("暂无训练数据，完成训练后将在此展示可视化图表")
        return

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import timedelta

    # 构建 DataFrame
    df = pd.DataFrame(records, columns=["ID", "日期", "动作类型", "次数", "平均分", "时长(秒)", "备注"])
    # 先将 bytes 类型的脏数据转为字符串，再做数值转换
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x, errors="replace") if isinstance(x, bytes) else x)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"])
    df["日期_str"] = df["日期"].dt.strftime("%Y-%m-%d %H:%M")
    df["日期_day"] = df["日期"].dt.strftime("%Y-%m-%d")
    df["时长(分钟)"] = pd.to_numeric(df["时长(秒)"], errors="coerce").fillna(0).astype(float) / 60
    # 将数值列转为 Python 原生类型，避免 plotly JSON 序列化报错
    df["次数"] = pd.to_numeric(df["次数"], errors="coerce").fillna(0).astype(int)
    df["平均分"] = pd.to_numeric(df["平均分"], errors="coerce").fillna(0).astype(float)
    df["时长(秒)"] = pd.to_numeric(df["时长(秒)"], errors="coerce").fillna(0).astype(float)

    # ===== 总览指标 =====
    stats = db.get_statistics(username=cur_user, user_id=cur_user_id)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总训练次数", f"{stats['total_sessions']} 次")
    c2.metric("总动作数", f"{stats['total_reps']} 个")
    c3.metric("总训练时长", f"{stats['total_duration'] / 60:.1f} 分钟")
    c4.metric("平均评分", f"{stats['avg_score']:.1f} 分")

    st.divider()

    # ===== 评分趋势 + 动作分布 =====
    col1, col2 = st.columns([2, 1])

    # plotly 隐藏工具栏
    plotly_config = {"displayModeBar": False}

    with col1:
        st.markdown("#### 📈 评分趋势")
        action_types = sorted(df["动作类型"].unique().tolist())
        selected_actions = st.multiselect(
            "选择动作", action_types, default=action_types[:1],
            key="trend_action_filter"
        )
        # 按天+动作类型聚合，取当天平均分
        df_filtered = df[df["动作类型"].isin(selected_actions)] if selected_actions else df
        df_trend = df_filtered.groupby(["日期_day", "动作类型"]).agg(
            {"平均分": "mean"}).reset_index()
        df_trend["平均分"] = df_trend["平均分"].round(1)
        fig_trend = px.line(
            df_trend, x="日期_day", y="平均分",
            color="动作类型", markers=True,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_trend.update_layout(
            xaxis_title="", yaxis_title="评分",
            xaxis=dict(type="category"),
            yaxis=dict(range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            height=350
        )
        st.plotly_chart(fig_trend, use_container_width=True, config=plotly_config)

    with col2:
        st.markdown("#### 🎯 动作分布")
        action_reps = df.groupby("动作类型")["次数"].sum().reset_index()
        fig_pie = px.pie(
            action_reps, values="次数", names="动作类型",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True, config=plotly_config)

    st.divider()

    # ===== 每日训练量 + 评分分布 =====
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 📊 每日训练量")
        daily = df.groupby("日期_day").agg({"次数": "sum"}).reset_index()
        daily.columns = ["日期", "总次数"]
        daily["总次数"] = daily["总次数"].astype(int)
        fig_daily = px.bar(
            daily, x="日期", y="总次数",
            color_discrete_sequence=["#667eea"]
        )
        fig_daily.update_layout(
            xaxis_title="", yaxis_title="动作次数",
            xaxis=dict(type="category"),
            yaxis=dict(dtick=max(1, daily["总次数"].max() // 5)),
            margin=dict(l=20, r=20, t=10, b=20),
            height=300
        )
        st.plotly_chart(fig_daily, use_container_width=True, config=plotly_config)

    with col4:
        st.markdown("#### 📉 评分分布")
        # 过滤掉0分的无效数据
        df_score = df[df["平均分"] > 0]
        fig_hist = px.histogram(
            df_score, x="平均分", nbins=10,
            color="动作类型",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_hist.update_layout(
            xaxis_title="评分", yaxis_title="频次",
            xaxis=dict(range=[0, 100]),
            yaxis=dict(dtick=1),
            margin=dict(l=20, r=20, t=10, b=20),
            height=300, bargap=0.1
        )
        st.plotly_chart(fig_hist, use_container_width=True, config=plotly_config)

    st.divider()

    # ===== 本周 vs 上周 =====
    st.markdown("#### 📅 本周 vs 上周")
    now = pd.Timestamp.now()
    this_week_start = (now - timedelta(days=now.weekday())).normalize()
    last_week_start = this_week_start - timedelta(days=7)

    tw = df[df["日期"] >= this_week_start]
    lw = df[(df["日期"] >= last_week_start) & (df["日期"] < this_week_start)]

    wc1, wc2, wc3 = st.columns(3)

    tw_reps = int(tw["次数"].sum()) if len(tw) > 0 else 0
    lw_reps = int(lw["次数"].sum()) if len(lw) > 0 else 0
    wc1.metric("本周动作数", f"{tw_reps} 个",
               delta=f"{tw_reps - lw_reps:+d}" if lw_reps > 0 else None)

    tw_score = float(tw["平均分"].mean()) if len(tw) > 0 else 0
    lw_score = float(lw["平均分"].mean()) if len(lw) > 0 else 0
    wc2.metric("本周均分", f"{tw_score:.1f}",
               delta=f"{tw_score - lw_score:+.1f}" if lw_score > 0 else None)

    tw_dur = float(tw["时长(秒)"].sum()) / 60 if len(tw) > 0 else 0
    lw_dur = float(lw["时长(秒)"].sum()) / 60 if len(lw) > 0 else 0
    wc3.metric("本周时长", f"{tw_dur:.1f} 分钟",
               delta=f"{tw_dur - lw_dur:+.1f} 分钟" if lw_dur > 0 else None)

    st.divider()

    # ===== 各动作雷达图（需至少2种动作数据）=====
    action_stats = df.groupby("动作类型").agg({
        "次数": "sum", "平均分": "mean",
        "时长(分钟)": "sum", "ID": "count"
    }).reset_index()
    action_stats.columns = ["动作类型", "总次数", "平均评分", "总时长", "训练次数"]

    if len(action_stats) >= 2:
        st.markdown("#### 🏋️ 各动作综合对比")
        categories = ["总次数", "平均评分", "训练频次", "总时长"]

        fig_radar = go.Figure()
        for _, row in action_stats.iterrows():
            vals = [
                float(row["总次数"] / max(action_stats["总次数"].max(), 1) * 100),
                float(row["平均评分"]),
                float(row["训练次数"] / max(action_stats["训练次数"].max(), 1) * 100),
                float(row["总时长"] / max(action_stats["总时长"].max(), 0.1) * 100),
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill='toself', name=row["动作类型"], opacity=0.7
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            margin=dict(l=60, r=60, t=30, b=30),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        st.divider()

    # ===== 数据导出 =====
    st.markdown("#### 💾 数据导出")
    export_df = df[["日期", "动作类型", "次数", "平均分", "时长(秒)"]].copy()
    export_df["日期"] = export_df["日期"].dt.strftime("%Y-%m-%d %H:%M:%S")
    csv_data = export_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 导出训练记录 (CSV)",
        data=csv_data,
        file_name=f"训练记录_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==================== 动作指南页面 ====================
def render_guide_page():
    """渲染动作指南页面"""
    st.markdown('<h1 class="main-title">📖 动作指南</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">标准动作要领 · 常见错误纠正 · 目标肌群说明</p>', unsafe_allow_html=True)

    db = init_database()
    guides = db.get_action_guides()

    if not guides:
        st.info("暂无动作指南数据")
        return

    # 动作图标映射
    action_icons = {"深蹲": "🦵", "俯卧撑": "💪", "卷腹": "🔥", "弓步蹲": "🏃"}
    difficulty_colors = {"初级": "#51cf66", "中级": "#ffd43b", "高级": "#ff6b6b"}

    # 顶部动作选择 tab
    action_names = [g[1] for g in guides]
    tabs = st.tabs([f"{action_icons.get(name, '🏋️')} {name}" for name in action_names])

    for tab, guide in zip(tabs, guides):
        _, action_type, description, key_points, common_mistakes, \
            target_muscles, difficulty, calories = guide

        with tab:
            # 标题行：难度 + 热量
            diff_color = difficulty_colors.get(difficulty, "#aaa")
            st.markdown(f"""
            <div style="display:flex; gap:1.5rem; align-items:center; margin-bottom:1rem;">
                <span style="background:{diff_color}22; color:{diff_color}; padding:4px 14px;
                      border-radius:20px; font-weight:600; border:1px solid {diff_color}44;">
                    {difficulty}
                </span>
                <span style="color:#aaa;">🔥 每次约消耗 {calories} kcal</span>
                <span style="color:#aaa;">💪 {target_muscles}</span>
            </div>
            """, unsafe_allow_html=True)

            # 动作简介
            st.markdown(f"**动作简介**：{description}")

            st.divider()

            col_good, col_bad = st.columns(2)

            # 动作要领
            with col_good:
                st.markdown("#### ✅ 动作要领")
                for i, point in enumerate(key_points.split("；"), 1):
                    point = point.strip()
                    if point:
                        st.markdown(f"""
                        <div class="good-box" style="margin:0.3rem 0; padding:0.6rem 1rem;">
                            <b>{i}.</b> {point}
                        </div>""", unsafe_allow_html=True)

            # 常见错误
            with col_bad:
                st.markdown("#### ❌ 常见错误")
                for i, mistake in enumerate(common_mistakes.split("；"), 1):
                    mistake = mistake.strip()
                    if mistake:
                        st.markdown(f"""
                        <div class="tip-box" style="margin:0.3rem 0; padding:0.6rem 1rem;">
                            <b>{i}.</b> {mistake}
                        </div>""", unsafe_allow_html=True)

            st.divider()

            # 评分规则（从 action_rules 表拉取）
            action_type_map = {"深蹲": "深蹲", "俯卧撑": "俯卧撑", "卷腹": "卷腹", "弓步蹲": "弓步蹲"}
            rules = db.get_action_rules(action_type_map.get(action_type, action_type))
            if rules:
                st.markdown("#### 📐 评分规则")
                for rule in rules:
                    _, _, rule_name, rule_value, prompt_text = rule
                    st.markdown(
                        f"- **{rule_name}**：阈值 `{rule_value:.0f}°` — {prompt_text}"
                    )


# ==================== 个人中心页面 ====================
def render_profile_page():
    """渲染个人中心页面"""
    st.markdown('<h1 class="main-title">👤 个人中心</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">个人信息管理 · BMI 健康评估 · 训练画像</p>', unsafe_allow_html=True)

    db = init_database()
    cur_user = get_current_username()
    cur_user_id = get_current_user_id()
    profile = db.get_user_profile(username=cur_user, user_id=cur_user_id)

    # ===== 信息填写表单 =====
    st.markdown("#### 📝 基本信息")

    # 从字典安全取值
    p_nickname = profile.get("nickname", "") if profile else ""
    p_gender = profile.get("gender", "男") if profile else "男"
    p_age = int(profile.get("age", 0) or 0) if profile else 20
    p_height = float(profile.get("height_cm", 0) or 0) if profile else 170.0
    p_weight = float(profile.get("weight_kg", 0) or 0) if profile else 65.0
    p_goal = profile.get("fitness_goal", "") if profile else ""
    goals_list = ["减脂塑形", "增肌增重", "体能提升", "康复训练", "日常保健"]

    with st.form("profile_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            nickname = st.text_input("昵称", value=p_nickname)
            gender = st.selectbox("性别", ["男", "女"],
                                  index=["男", "女"].index(p_gender) if p_gender in ["男", "女"] else 0)
        with col2:
            age = st.number_input("年龄", min_value=1, max_value=120,
                                  value=p_age if 1 <= p_age <= 120 else 20)
            height_cm = st.number_input("身高 (cm)", min_value=50.0, max_value=250.0, step=0.1,
                                        value=p_height if 50 <= p_height <= 250 else 170.0)
        with col3:
            weight_kg = st.number_input("体重 (kg)", min_value=20.0, max_value=300.0, step=0.1,
                                        value=p_weight if 20 <= p_weight <= 300 else 65.0)
            fitness_goal = st.selectbox("健身目标", goals_list,
                                        index=goals_list.index(p_goal) if p_goal in goals_list else 0)

        submitted = st.form_submit_button("💾 保存信息", use_container_width=True, type="primary")
        if submitted:
            db.save_user_profile(
                nickname,
                gender,
                age,
                height_cm,
                weight_kg,
                fitness_goal,
                username=cur_user,
                user_id=cur_user_id,
            )
            st.toast("✅ 信息保存成功！")
            st.rerun()

    if not profile:
        st.info("请先填写个人信息并保存")
        return

    st.divider()

    # ===== BMI 健康评估 =====
    nickname_v = p_nickname
    gender_v = p_gender
    age_v = p_age
    height_v = p_height if p_height > 0 else 170
    weight_v = p_weight if p_weight > 0 else 65
    goal_v = p_goal
    created_v = profile.get("created_at", "") if profile else ""

    bmi = weight_v / ((height_v / 100) ** 2) if height_v > 0 else 0

    if bmi < 18.5:
        bmi_label, bmi_color, bmi_advice = "偏瘦", "#54a0ff", "建议增加营养摄入，配合增肌训练"
    elif bmi < 24:
        bmi_label, bmi_color, bmi_advice = "正常", "#51cf66", "体重健康，继续保持良好习惯"
    elif bmi < 28:
        bmi_label, bmi_color, bmi_advice = "偏胖", "#ffd43b", "建议增加有氧运动，控制饮食热量"
    else:
        bmi_label, bmi_color, bmi_advice = "肥胖", "#ff6b6b", "建议制定系统减脂计划，注意饮食管理"

    st.markdown("#### 📊 BMI 健康评估")

    bmi_col1, bmi_col2, bmi_col3 = st.columns(3)
    bmi_col1.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:{bmi_color};">{bmi:.1f}</div>
        <div class="metric-label">BMI 指数</div></div>""", unsafe_allow_html=True)
    bmi_col2.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:{bmi_color}; font-size:1.8rem;">{bmi_label}</div>
        <div class="metric-label">体型评估</div></div>""", unsafe_allow_html=True)
    bmi_col3.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#aaa; font-size:1.1rem;">{bmi_advice}</div>
        <div class="metric-label">健康建议</div></div>""", unsafe_allow_html=True)

    # BMI 标尺
    bmi_display = min(max(bmi, 14), 36)
    bmi_pct = (bmi_display - 14) / (36 - 14) * 100
    st.markdown(f"""
    <div style="margin:1.5rem 0;">
        <div style="position:relative; height:30px; border-radius:15px; overflow:hidden;
             background: linear-gradient(90deg, #54a0ff 0%, #51cf66 25%, #51cf66 45%,
             #ffd43b 45%, #ffd43b 64%, #ff6b6b 64%, #ff6b6b 100%);">
            <div style="position:absolute; left:{bmi_pct}%; top:-2px;
                 width:4px; height:34px; background:white; border-radius:2px;
                 box-shadow: 0 0 8px rgba(255,255,255,0.8);"></div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:4px; font-size:0.75rem; color:#888;">
            <span>偏瘦 &lt;18.5</span><span>正常 18.5-24</span>
            <span>偏胖 24-28</span><span>肥胖 &gt;28</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ===== 训练画像 =====
    st.markdown("#### 🏅 训练画像")
    stats = db.get_statistics(username=cur_user, user_id=cur_user_id)

    # 计算训练天数
    records = db.get_training_records(limit=9999, username=cur_user, user_id=cur_user_id)
    if records:
        import pandas as pd
        df_r = pd.DataFrame(records, columns=["ID", "日期", "动作类型", "次数", "平均分", "时长", "备注"])
        df_r["日期"] = pd.to_datetime(df_r["日期"])
        training_days = df_r["日期"].dt.date.nunique()
        first_day = df_r["日期"].min().strftime("%Y-%m-%d")
        favorite_action = df_r.groupby("动作类型")["次数"].sum().idxmax()
        best_score = float(pd.to_numeric(df_r["平均分"], errors="coerce").max())
    else:
        training_days, first_day, favorite_action, best_score = 0, "—", "—", 0

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("训练天数", f"{training_days} 天")
    p2.metric("最爱动作", favorite_action)
    p3.metric("最高评分", f"{best_score:.1f}")
    p4.metric("首次训练", first_day)

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(f"""
        **个人信息卡**
        - 昵称：{nickname_v or '未设置'}
        - 性别：{gender_v}　|　年龄：{age_v} 岁
        - 身高：{height_v} cm　|　体重：{weight_v} kg
        - 健身目标：{goal_v}
        - 注册时间：{created_v or '—'}
        """)
    with info_col2:
        st.markdown(f"""
        **训练统计**
        - 总训练次数：{stats['total_sessions']} 次
        - 总动作完成：{stats['total_reps']} 个
        - 总训练时长：{stats['total_duration'] / 60:.1f} 分钟
        - 综合平均分：{stats['avg_score']:.1f} 分
        """)


# ==================== 训练计划页面 ====================
def render_plan_page():
    """渲染训练计划页面"""
    st.markdown('<h1 class="main-title">📋 训练计划</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">制定目标 · 追踪进度 · 自动同步训练记录</p>', unsafe_allow_html=True)

    db = init_database()
    cur_user = get_current_username()
    cur_user_id = get_current_user_id()
    today = datetime.now().strftime("%Y-%m-%d")

    # 自动同步今日计划与训练记录
    db.sync_plans_with_records(username=cur_user, plan_date=today, user_id=cur_user_id)

    # ===== 新建计划 =====
    with st.expander("➕ 新建训练计划", expanded=False):
        with st.form("add_plan_form"):
            pc1, pc2, pc3, pc4 = st.columns(4)
            with pc1:
                plan_action = st.selectbox("动作类型", ["深蹲", "俯卧撑", "卷腹", "弓步蹲"])
            with pc2:
                plan_reps = st.number_input("目标次数", min_value=1, max_value=500, value=20)
            with pc3:
                plan_score = st.number_input("目标评分", min_value=0.0, max_value=100.0, value=60.0, step=5.0)
            with pc4:
                plan_date = st.date_input("计划日期", value=datetime.now())
            plan_btn = st.form_submit_button("添加计划", use_container_width=True, type="primary")
        if plan_btn:
            db.add_plan(
                username=cur_user,
                action_type=plan_action,
                target_reps=plan_reps,
                target_score=plan_score,
                plan_date=plan_date.strftime("%Y-%m-%d"),
                user_id=cur_user_id,
            )
            st.toast("✅ 计划已添加")
            st.rerun()

    st.divider()

    # ===== 今日计划 =====
    st.markdown(f"#### 📅 今日计划（{today}）")
    today_plans = db.get_plans(username=cur_user, plan_date=today, user_id=cur_user_id)

    if not today_plans:
        st.info("今日暂无训练计划，点击上方「新建训练计划」创建")
    else:
        # 今日总览
        total_plans = len(today_plans)
        completed = sum(1 for p in today_plans if p[6] == 1)
        st.progress(completed / total_plans if total_plans > 0 else 0,
                     text=f"今日完成进度：{completed}/{total_plans} 项")

        for plan in today_plans:
            plan_id, _, action_type, target_reps, target_score, \
                p_date, is_done, actual_reps, actual_score = plan

            reps_pct = min(actual_reps / target_reps, 1.0) if target_reps > 0 else 0
            score_ok = actual_score >= target_score if actual_score > 0 else False

            if is_done:
                status_icon = "✅"
                card_class = "good-box"
            elif actual_reps > 0:
                status_icon = "🔄"
                card_class = "tip-box"
            else:
                status_icon = "⏳"
                card_class = "tip-box"

            st.markdown(f"""
            <div class="{card_class}" style="padding:1rem; margin:0.5rem 0;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.1rem; font-weight:600;">
                        {status_icon} {action_type}
                    </span>
                    <span style="font-size:0.85rem;">
                        次数：{actual_reps}/{target_reps}
                        评分：{actual_score:.1f}/{target_score:.0f}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 次数进度条
            st.progress(reps_pct, text=f"次数完成 {reps_pct * 100:.0f}%")

    st.divider()

    # ===== 历史计划 =====
    st.markdown("#### 📜 历史计划")
    all_plans = db.get_plans(username=cur_user, user_id=cur_user_id)
    # 过滤掉今日的
    history_plans = [p for p in all_plans if p[5] != today]

    if not history_plans:
        st.info("暂无历史计划记录")
    else:
        import pandas as pd
        df_plans = pd.DataFrame(history_plans,
                                columns=["ID", "用户", "动作类型", "目标次数", "目标评分",
                                         "计划日期", "是否完成", "实际次数", "实际评分"])
        df_plans["是否完成"] = df_plans["是否完成"].map({1: "✅ 已完成", 0: "❌ 未完成"})
        df_display = df_plans[["计划日期", "动作类型", "目标次数", "目标评分",
                               "实际次数", "实际评分", "是否完成"]]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # 完成率统计
        total_hist = len(history_plans)
        done_hist = sum(1 for p in history_plans if p[6] == 1)
        rate = done_hist / total_hist * 100 if total_hist > 0 else 0
        st.metric("历史计划完成率", f"{rate:.0f}%（{done_hist}/{total_hist}）")

    st.divider()

    # ===== 删除计划 =====
    if all_plans:
        with st.expander("🗑️ 删除计划"):
            plan_options = {
                f"[{p[5]}] {p[2]} - 目标{p[3]}次 (ID:{p[0]})": p[0]
                for p in all_plans
            }
            del_plan_sel = st.selectbox("选择要删除的计划", list(plan_options.keys()))
            if st.button("删除该计划"):
                db.delete_plan(plan_options[del_plan_sel])
                st.toast("✅ 计划已删除")
                st.rerun()


# ==================== 登录页面 ====================
def render_ai_report_page():
    """渲染 AI 专属私教页面"""
    st.markdown('<h1 class="main-title">🤖 AI 专属私教</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">生成训练报告、导出 PDF，并围绕历史报告进行专属追问</p>',
                unsafe_allow_html=True)

    db = init_database()
    ai_store = init_ai_store()
    cur_user = get_current_username()
    cur_user_id = get_current_user_id()
    settings = ai_store.get_ai_settings()
    stats = db.get_statistics(username=cur_user, user_id=cur_user_id)
    profile = db.get_user_profile(username=cur_user, user_id=cur_user_id)
    plans = db.get_plans(username=cur_user, user_id=cur_user_id)
    ai_enabled = bool(settings.get("enabled", 0))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("累计训练次数", f"{stats.get('total_sessions', 0)} 次")
    m2.metric("累计动作数", f"{stats.get('total_reps', 0)} 个")
    m3.metric("平均评分", f"{stats.get('avg_score', 0):.1f}")
    m4.metric("训练计划数", f"{len(plans)} 项")

    st.caption(
        f"当前接入：{settings.get('provider_name', '未配置')} / {settings.get('model_name', '未配置')} | "
        f"API Key：{mask_secret(settings.get('api_key', ''))}"
    )
    st.caption("支持生成固定格式训练报告、导出 PDF，以及基于指定报告进行 AI 私教问答。")

    if not ai_enabled:
        st.warning("AI 功能尚未启用。请管理员先到后台完成模型接入设置。")
    if not profile:
        st.info("当前用户尚未完善个人资料。可以先生成报告，但分析会缺少身体数据和目标信息。")
    if stats.get("total_sessions", 0) <= 0:
        st.info("当前用户还没有训练记录，完成训练后再生成 AI 报告。")

    reports = ai_store.get_ai_reports(username=cur_user, user_id=cur_user_id, limit=20)
    report_ids = [item["report_id"] for item in reports]
    if report_ids:
        if st.session_state.get("selected_ai_report_id") not in report_ids:
            st.session_state["selected_ai_report_id"] = report_ids[0]
        if st.session_state.get("selected_ai_chat_report_id") not in report_ids:
            st.session_state["selected_ai_chat_report_id"] = st.session_state.get("selected_ai_report_id")

    label_map = {
        item["report_id"]: (
            f"{item['report_time']} | {item['provider_name']} / {item['model_name']} "
            f"| {item['record_count']} 条记录"
        )
        for item in reports
    }

    report_tab, chat_tab = st.tabs(["训练报告", "专属问答"])

    with report_tab:
        scope_options = {
            "最近 10 条训练记录": 10,
            "最近 30 条训练记录": 30,
            "最近 50 条训练记录": 50,
            "最近 100 条训练记录": 100,
        }
        selected_scope = st.selectbox("分析范围", list(scope_options.keys()), index=1)

        generate_disabled = (not ai_enabled) or stats.get("total_sessions", 0) <= 0
        if st.button("生成训练报告", type="primary", use_container_width=True, disabled=generate_disabled):
            with st.spinner("正在汇总训练数据并生成训练报告..."):
                try:
                    result = generate_ai_training_report(
                        db,
                        username=cur_user,
                        user_id=cur_user_id,
                        max_records=scope_options[selected_scope],
                        ai_store=ai_store,
                    )
                    report_id = ai_store.save_ai_report(
                        username=cur_user,
                        user_id=cur_user_id,
                        provider_name=result["provider_name"],
                        model_name=result["model_name"],
                        record_count=result["record_count"],
                        source_summary=result["source_summary"],
                        report_json=result["report"],
                        raw_response=result["raw_response_text"],
                    )
                    st.session_state["selected_ai_report_id"] = report_id
                    st.session_state["selected_ai_chat_report_id"] = report_id
                    st.session_state["selected_ai_chat_session_id"] = None
                    st.toast("训练报告已生成")
                    st.rerun()
                except AIReportError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"生成失败：{exc}")

        st.divider()
        st.markdown("#### 历史报告")
        if not reports:
            st.info("还没有训练报告历史。")
        else:
            selected_report_id = st.selectbox(
                "选择历史报告",
                report_ids,
                format_func=lambda rid: label_map.get(rid, str(rid)),
                key="selected_ai_report_id",
            )

            detail = ai_store.get_ai_report(selected_report_id, username=cur_user, user_id=cur_user_id)
            if detail is None:
                st.error("未找到所选报告。")
            else:
                action_col1, action_col2, action_col3 = st.columns([1, 1, 4])
                with action_col1:
                    try:
                        pdf_bytes = build_report_pdf_bytes(detail)
                        st.download_button(
                            "导出 PDF",
                            data=pdf_bytes,
                            file_name=build_ai_report_pdf_filename(detail),
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except AICoachError as exc:
                        st.warning(str(exc))
                    except Exception as exc:
                        st.warning(f"PDF 导出暂不可用：{exc}")
                with action_col2:
                    if st.button("删除报告", use_container_width=True, type="secondary"):
                        st.session_state["pending_delete_ai_report_id"] = selected_report_id
                with action_col3:
                    st.caption("当前导出的是所选历史报告的完整 PDF 版本。")

                if st.session_state.get("pending_delete_ai_report_id") == selected_report_id:
                    st.warning("确认删除当前历史报告？删除后，该报告及其关联聊天都会被移除。")
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        if st.button("确认删除报告", use_container_width=True, type="primary"):
                            deleted = ai_store.delete_ai_report(
                                selected_report_id,
                                username=cur_user,
                                user_id=cur_user_id,
                            )
                            st.session_state["pending_delete_ai_report_id"] = None
                            if deleted:
                                st.toast("历史报告已删除")
                            st.rerun()
                    with rc2:
                        if st.button("取消删除报告", use_container_width=True):
                            st.session_state["pending_delete_ai_report_id"] = None
                            st.rerun()

                render_ai_report_content(detail)

    with chat_tab:
        st.markdown("#### AI 私教追问")
        if not ai_enabled:
            st.warning("AI 功能尚未启用，暂时无法进行专属问答。")
        elif not reports:
            st.info("请先生成至少一份训练报告，再围绕报告向 AI 私教追问。")
        else:
            selected_chat_report_id = st.selectbox(
                "选择要追问的报告",
                report_ids,
                format_func=lambda rid: label_map.get(rid, str(rid)),
                key="selected_ai_chat_report_id",
            )
            chat_detail = ai_store.get_ai_report(
                selected_chat_report_id,
                username=cur_user,
                user_id=cur_user_id,
            )
            if chat_detail is None:
                st.error("未找到所选报告。")
            else:
                summary = str(chat_detail.get("report_json", {}).get("summary") or "").strip()
                if summary:
                    st.markdown(f'<div class="good-box"><b>当前报告摘要：</b>{summary}</div>', unsafe_allow_html=True)

                sessions = ai_store.get_chat_sessions(
                    selected_chat_report_id,
                    username=cur_user,
                    user_id=cur_user_id,
                    limit=30,
                )
                if not sessions:
                    new_session_id = ai_store.create_chat_session(
                        selected_chat_report_id,
                        username=cur_user,
                        user_id=cur_user_id,
                    )
                    st.session_state["selected_ai_chat_session_id"] = new_session_id
                    st.rerun()

                session_ids = [item["session_id"] for item in sessions]
                pending_session_id = st.session_state.get("pending_select_ai_chat_session_id")
                if pending_session_id in session_ids:
                    st.session_state["selected_ai_chat_session_id"] = pending_session_id
                elif st.session_state.get("selected_ai_chat_session_id") not in session_ids:
                    st.session_state["selected_ai_chat_session_id"] = session_ids[0]
                st.session_state["pending_select_ai_chat_session_id"] = None

                left_col, right_col = st.columns([1.1, 3.4], gap="large")
                with left_col:
                    st.markdown("##### 你的聊天")
                    st.caption("每份报告都可以开启多个独立聊天窗口。")
                    st.radio(
                        "聊天记录",
                        session_ids,
                        format_func=lambda sid: format_ai_chat_session_label(
                            next(item for item in sessions if item["session_id"] == sid)
                        ),
                        key="selected_ai_chat_session_id",
                        label_visibility="collapsed",
                    )
                    if st.button("开启新聊天", use_container_width=True):
                        new_session_id = ai_store.create_chat_session(
                            selected_chat_report_id,
                            username=cur_user,
                            user_id=cur_user_id,
                        )
                        st.session_state["pending_select_ai_chat_session_id"] = new_session_id
                        st.session_state["pending_delete_ai_chat_session_id"] = None
                        st.toast("已开启新的聊天窗口")
                        st.rerun()
                    if st.button("删除当前聊天", use_container_width=True, type="secondary"):
                        st.session_state["pending_delete_ai_chat_session_id"] = st.session_state.get(
                            "selected_ai_chat_session_id"
                        )

                    pending_session_id = st.session_state.get("pending_delete_ai_chat_session_id")
                    if pending_session_id in session_ids:
                        st.warning("确认删除当前聊天？该聊天下的所有问答都会被移除。")
                        dc1, dc2 = st.columns(2)
                        with dc1:
                            if st.button("确认删除聊天", use_container_width=True, type="primary"):
                                ai_store.delete_chat_session(
                                    pending_session_id,
                                    username=cur_user,
                                    user_id=cur_user_id,
                                )
                                st.session_state["pending_delete_ai_chat_session_id"] = None
                                st.toast("聊天记录已删除")
                                st.rerun()
                        with dc2:
                            if st.button("取消删除聊天", use_container_width=True):
                                st.session_state["pending_delete_ai_chat_session_id"] = None
                                st.rerun()

                with right_col:
                    st.caption("问答会围绕当前所选报告展开，不会脱离该报告随意推断。")
                    active_session_id = st.session_state.get("selected_ai_chat_session_id")
                    messages = ai_store.get_report_chat_messages(
                        selected_chat_report_id,
                        username=cur_user,
                        user_id=cur_user_id,
                        limit=80,
                        session_id=active_session_id,
                    )
                    render_ai_chat_messages(messages)

                    question = st.chat_input(
                        "围绕当前报告继续追问，例如：下周训练怎么安排更合理？",
                        key=f"ai_coach_chat_input_{selected_chat_report_id}_{active_session_id}",
                    )

                    if question:
                        with st.spinner("AI 私教正在阅读报告并组织回答..."):
                            try:
                                result = generate_ai_coach_reply(
                                    chat_detail,
                                    question,
                                    ai_store=ai_store,
                                    history_limit=12,
                                    session_id=active_session_id,
                                )
                                ai_store.save_report_chat_message(
                                    selected_chat_report_id,
                                    username=cur_user,
                                    user_id=cur_user_id,
                                    role="user",
                                    content=question,
                                    session_id=active_session_id,
                                )
                                ai_store.save_report_chat_message(
                                    selected_chat_report_id,
                                    username=cur_user,
                                    user_id=cur_user_id,
                                    role="assistant",
                                    content=result["answer"],
                                    session_id=active_session_id,
                                )
                                st.toast("AI 私教已回复")
                                st.rerun()
                            except AICoachError as exc:
                                st.error(str(exc))
                            except Exception as exc:
                                st.error(f"问答失败：{exc}")


def render_admin_ai_page():
    """渲染管理员 AI 设置页面"""
    st.markdown('<h1 class="main-title">🤖 AI 模型接入设置</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">配置模型接口、提示词和报告历史查看</p>', unsafe_allow_html=True)

    ai_store = init_ai_store()
    settings = ai_store.get_ai_settings()
    current_provider = get_provider_option(provider_name=settings.get("provider_name"))

    def ensure_admin_ai_draft():
        loaded_at = st.session_state.get("admin_ai_draft_loaded_at")
        updated_at = settings.get("updated_at", "")
        if loaded_at == updated_at and "admin_ai_provider_key" in st.session_state:
            return

        provider = get_provider_option(provider_name=settings.get("provider_name"))
        model_name = (settings.get("model_name") or "").strip()
        base_url = (settings.get("base_url") or "").strip()
        if not base_url:
            base_url = provider_default_base_url(provider["key"])

        st.session_state["admin_ai_enabled"] = bool(settings.get("enabled", 0))
        st.session_state["admin_ai_provider_key"] = provider["key"]
        st.session_state["admin_ai_base_url"] = base_url
        st.session_state["admin_ai_model_name_input"] = model_name
        st.session_state["admin_ai_api_key"] = ""
        st.session_state["admin_ai_temperature"] = float(settings.get("temperature", 0.2) or 0.2)
        st.session_state["admin_ai_timeout_sec"] = int(settings.get("timeout_sec", 60) or 60)
        st.session_state["admin_ai_system_prompt"] = settings.get("system_prompt", "")
        st.session_state["admin_ai_available_models"] = []
        st.session_state["admin_ai_model_picker"] = model_name if model_name else "手动输入"
        st.session_state["admin_ai_previous_provider_key"] = provider["key"]
        st.session_state["admin_ai_draft_loaded_at"] = updated_at

    def on_admin_ai_provider_change():
        provider_key = st.session_state.get("admin_ai_provider_key", "custom_openai")
        previous_key = st.session_state.get("admin_ai_previous_provider_key")
        current_base_url = (st.session_state.get("admin_ai_base_url") or "").strip()
        previous_default = provider_default_base_url(previous_key) if previous_key else ""
        next_default = provider_default_base_url(provider_key)

        if not current_base_url or current_base_url == previous_default:
            st.session_state["admin_ai_base_url"] = next_default

        current_model_name = (st.session_state.get("admin_ai_model_name_input") or "").strip()
        st.session_state["admin_ai_available_models"] = []
        st.session_state["admin_ai_model_picker"] = current_model_name if current_model_name else "手动输入"
        st.session_state["admin_ai_previous_provider_key"] = provider_key

    ensure_admin_ai_draft()

    def sync_admin_ai_model_state():
        available_models = st.session_state.get("admin_ai_available_models", [])
        if not available_models:
            return

        current_model_name = (st.session_state.get("admin_ai_model_name_input") or "").strip()
        model_picker = st.session_state.get("admin_ai_model_picker", "手动输入")
        model_options = ["手动输入"] + available_models

        if model_picker not in model_options:
            st.session_state["admin_ai_model_picker"] = "手动输入"
            model_picker = "手动输入"

        if current_model_name and current_model_name not in available_models:
            if model_picker != "手动输入":
                st.session_state["admin_ai_model_picker"] = "手动输入"
            return

        if model_picker != "手动输入" and model_picker in available_models and current_model_name != model_picker:
            st.session_state["admin_ai_model_name_input"] = model_picker

    sync_admin_ai_model_state()

    c1, c2, c3 = st.columns(3)
    c1.metric("启用状态", "已启用" if settings.get("enabled", 0) else "未启用")
    c2.metric("服务商", current_provider["label"] or "未配置")
    c3.metric("模型名", settings.get("model_name", "") or "未配置")

    st.caption("当前支持自定义（兼容 OpenAI）、Google AI Studio、OpenAI、Claude。API Key 留空表示保持原值。")
    st.caption(f"当前已保存 API Key：{mask_secret(settings.get('api_key', ''))}")

    st.checkbox("启用 AI 报告功能", key="admin_ai_enabled")

    provider_keys = [item["key"] for item in PROVIDER_OPTIONS]
    if st.session_state.get("admin_ai_provider_key") not in provider_keys:
        st.session_state["admin_ai_provider_key"] = provider_keys[0]
    st.selectbox(
        "服务商",
        provider_keys,
        format_func=lambda key: get_provider_option(provider_key=key)["label"],
        key="admin_ai_provider_key",
        on_change=on_admin_ai_provider_change,
    )

    selected_provider = get_provider_option(provider_key=st.session_state.get("admin_ai_provider_key"))
    if selected_provider["default_base_url"]:
        st.caption(
            f"认证方式：{selected_provider['auth_hint']}。默认 Base URL：{selected_provider['default_base_url']}。"
        )
    else:
        st.caption(f"认证方式：{selected_provider['auth_hint']}。自定义模式请填写兼容接口的完整 Base URL。")

    st.text_input(
        "Base URL",
        key="admin_ai_base_url",
        placeholder=selected_provider["default_base_url"] or "例如：https://api.deepseek.com",
    )

    st.text_input("API Key", key="admin_ai_api_key", type="password", placeholder="留空则保持不变")

    model_col, fetch_col = st.columns([3, 1])
    with model_col:
        st.text_input(
            "模型名称",
            key="admin_ai_model_name_input",
            placeholder="可手动输入，或点击右侧按钮拉取模型列表",
        )
    with fetch_col:
        fetch_models_btn = st.button("拉取模型", use_container_width=True)

    if fetch_models_btn:
        draft_provider_key = st.session_state.get("admin_ai_provider_key", "custom_openai")
        draft_api_key = (st.session_state.get("admin_ai_api_key") or "").strip() or settings.get("api_key", "")
        draft_base_url = (st.session_state.get("admin_ai_base_url") or "").strip()
        temp_settings = {
            "enabled": 1,
            "provider_name": provider_label(draft_provider_key),
            "base_url": draft_base_url,
            "api_key": draft_api_key,
            "model_name": st.session_state.get("admin_ai_model_name_input", ""),
            "temperature": st.session_state.get("admin_ai_temperature", 0.2),
            "timeout_sec": st.session_state.get("admin_ai_timeout_sec", 60),
        }
        with st.spinner("正在拉取可用模型..."):
            try:
                client = MultiProviderAIClient(temp_settings)
                models = client.fetch_models()
                st.session_state["admin_ai_available_models"] = models
                if models:
                    current_model_name = (st.session_state.get("admin_ai_model_name_input") or "").strip()
                    selected_model_name = current_model_name if current_model_name in models else models[0]
                    st.session_state["admin_ai_model_picker"] = selected_model_name
                    st.toast(f"已拉取 {len(models)} 个模型")
                    st.rerun()
                else:
                    st.session_state["admin_ai_model_picker"] = "手动输入"
                    st.warning("接口返回了空模型列表，请检查服务商权限，或手动填写模型名称。")
            except (AIConfigError, AIResponseError) as exc:
                st.session_state["admin_ai_available_models"] = []
                st.error(f"模型拉取失败：{exc}")
            except Exception as exc:
                st.session_state["admin_ai_available_models"] = []
                st.error(f"模型拉取失败：{exc}")

    available_models = st.session_state.get("admin_ai_available_models", [])
    if available_models:
        current_model_name = (st.session_state.get("admin_ai_model_name_input") or "").strip()
        model_options = ["手动输入"] + available_models
        desired_picker = current_model_name if current_model_name in available_models else "手动输入"
        if st.session_state.get("admin_ai_model_picker") not in model_options:
            st.session_state["admin_ai_model_picker"] = desired_picker

        picked_model = st.selectbox(
            "可用模型",
            model_options,
            key="admin_ai_model_picker",
            help="选择后会自动写入上方的模型名称输入框。",
        )
        st.caption(f"当前已拉取 {len(available_models)} 个模型，可继续手动修改模型名称。")

    s1, s2 = st.columns(2)
    with s1:
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            key="admin_ai_temperature",
        )
    with s2:
        st.number_input(
            "超时时间(秒)",
            min_value=10,
            max_value=300,
            step=5,
            key="admin_ai_timeout_sec",
        )

    st.text_area(
        "补充系统提示词",
        key="admin_ai_system_prompt",
        height=180,
        placeholder="可选。比如要求报告更简洁、强调计划执行、控制语气等。",
    )

    save_ai_btn = st.button("保存 AI 设置", use_container_width=True, type="primary")
    if save_ai_btn:
        ai_store.save_ai_settings(
            enabled=st.session_state.get("admin_ai_enabled", False),
            provider_name=provider_label(st.session_state.get("admin_ai_provider_key", "custom_openai")),
            base_url=st.session_state.get("admin_ai_base_url", ""),
            api_key=st.session_state.get("admin_ai_api_key", ""),
            model_name=st.session_state.get("admin_ai_model_name_input", ""),
            system_prompt=st.session_state.get("admin_ai_system_prompt", ""),
            temperature=st.session_state.get("admin_ai_temperature", 0.2),
            timeout_sec=st.session_state.get("admin_ai_timeout_sec", 60),
        )
        st.toast("✅ AI 设置已保存")
        st.rerun()

    st.divider()
    st.markdown("#### 最近生成的 AI 报告")
    recent_reports = ai_store.get_ai_reports(limit=20)
    if recent_reports:
        import pandas as pd
        df_reports = pd.DataFrame(recent_reports)
        df_reports.columns = ["报告ID", "用户名", "生成时间", "服务商", "模型名", "分析记录数"]
        st.dataframe(df_reports, use_container_width=True, hide_index=True)
    else:
        st.info("暂无 AI 报告记录")


def render_login_page():
    """渲染登录页面"""
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; margin-top:8vh;">
        <h1 class="main-title" style="font-size:2.5rem;">🏋️ 健身动作智能识别系统</h1>
        <p style="color:#888; margin-bottom:2rem;">基于 MediaPipe 与轻量级深度学习模型 | 本地化运行 · 隐私安全</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_form, col_r = st.columns([1, 1.5, 1])
    with col_form:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e); border-radius:16px;
             padding:2rem; border:1px solid #2a2a4a; margin-top:1rem;">
            <h3 style="text-align:center; color:#00d4ff; margin-bottom:1.5rem;">用户登录</h3>
        </div>
        """, unsafe_allow_html=True)

        # 身份选择
        login_role = st.radio("选择登录身份", ["👤 用户登录", "🔑 管理员登录"],
                              index=0, horizontal=True)

        login_tab, register_tab = st.tabs(["🔑 登录", "📝 注册"])

        with login_tab:
            with st.form("login_form"):
                username = st.text_input("账号", placeholder="请输入用户名")
                password = st.text_input("密码", type="password", placeholder="请输入密码")
                login_btn = st.form_submit_button("登 录", use_container_width=True, type="primary")

            if login_btn:
                if not username or not password:
                    st.error("请输入账号和密码")
                else:
                    db = init_database()
                    user = db.verify_user(username, password)
                    if not user:
                        st.error("账号或密码错误")
                    elif login_role == "🔑 管理员登录" and user[2] != "admin":
                        st.error("该账号不是管理员，请切换到用户登录")
                    elif login_role == "👤 用户登录" and user[2] == "admin":
                        st.error("管理员请切换到管理员登录入口")
                    else:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user[0]
                        st.session_state.username = user[1]
                        st.session_state.user_role = user[2]
                        st.rerun()

        with register_tab:
            with st.form("register_form"):
                new_user = st.text_input("设置用户名", placeholder="3-20个字符")
                new_pwd = st.text_input("设置密码", type="password", placeholder="至少6位")
                new_pwd2 = st.text_input("确认密码", type="password", placeholder="再次输入密码")
                reg_btn = st.form_submit_button("注 册", use_container_width=True)

            if reg_btn:
                if not new_user or not new_pwd:
                    st.error("请填写完整信息")
                elif len(new_user) < 3:
                    st.error("用户名至少3个字符")
                elif len(new_pwd) < 6:
                    st.error("密码至少6位")
                elif new_pwd != new_pwd2:
                    st.error("两次密码不一致")
                else:
                    db = init_database()
                    if db.add_user(new_user, new_pwd, role="user"):
                        st.success("✅ 注册成功！请切换到登录页面登录")
                    else:
                        st.error("用户名已存在")

        st.markdown("""
        <div style="text-align:center; color:#666; font-size:0.8rem; margin-top:1.5rem;">
            如果遗忘密码，请及时联系管理员
        </div>
        """, unsafe_allow_html=True)


# ==================== 管理员后台 ====================
def render_admin_page():
    """渲染管理员后台页面"""
    st.markdown('<h1 class="main-title">🔧 系统管理后台</h1>', unsafe_allow_html=True)

    db = init_database()
    admin_tab1, admin_tab2, admin_tab3 = st.tabs(["👥 用户管理", "📋 记录管理", "🗄️ 数据库概览"])

    # ===== 用户管理 =====
    with admin_tab1:
        st.markdown("#### 👥 用户列表")
        users = db.get_all_users()
        if users:
            import pandas as pd
            df_users = pd.DataFrame(users, columns=["ID", "用户名", "角色", "注册时间"])
            st.dataframe(df_users, use_container_width=True, hide_index=True)

            st.divider()

            # 添加用户
            st.markdown("#### ➕ 添加用户")
            with st.form("add_user_form"):
                au_col1, au_col2, au_col3 = st.columns(3)
                with au_col1:
                    au_name = st.text_input("用户名")
                with au_col2:
                    au_pwd = st.text_input("密码", type="password")
                with au_col3:
                    au_role = st.selectbox("角色", ["user", "admin"])
                au_btn = st.form_submit_button("添加", use_container_width=True)
            if au_btn:
                if au_name and au_pwd:
                    if db.add_user(au_name, au_pwd, au_role):
                        st.toast("✅ 用户添加成功")
                        st.rerun()
                    else:
                        st.error("用户名已存在")

            # 重置密码
            st.divider()
            st.markdown("#### 🔒 重置用户密码")
            normal_users = [u for u in users if u[2] != "admin"]
            if normal_users:
                pwd_options = {f"{u[1]} (ID:{u[0]})": u[0] for u in normal_users}
                pwd_sel = st.selectbox("选择用户", list(pwd_options.keys()), key="pwd_reset_user")
                with st.form("reset_pwd_form"):
                    new_pwd = st.text_input("新密码", type="password", placeholder="至少6位")
                    new_pwd2 = st.text_input("确认新密码", type="password", placeholder="再次输入")
                    reset_btn = st.form_submit_button("重置密码", use_container_width=True)
                if reset_btn:
                    if not new_pwd or len(new_pwd) < 6:
                        st.error("密码至少6位")
                    elif new_pwd != new_pwd2:
                        st.error("两次密码不一致")
                    else:
                        db.reset_password(pwd_options[pwd_sel], new_pwd)
                        st.toast("✅ 密码已重置")
            else:
                st.info("暂无普通用户")

            # 删除用户
            st.divider()
            st.markdown("#### 🗑️ 删除用户")
            if normal_users:
                del_options = {f"{u[1]} (ID:{u[0]})": u[0] for u in normal_users}
                del_sel = st.selectbox("选择要删除的用户", list(del_options.keys()))
                if st.button("删除该用户", type="secondary"):
                    db.delete_user(del_options[del_sel])
                    st.toast("✅ 已删除")
                    st.rerun()
            else:
                st.info("暂无可删除的普通用户")

    # ===== 记录管理 =====
    with admin_tab2:
        st.markdown("#### 📋 训练记录管理")

        # 按用户筛选
        all_users = db.get_all_users()
        user_options = ["全部用户"] + [u[1] for u in all_users]
        selected_user = st.selectbox("筛选用户", user_options, key="admin_rec_user")
        filter_username = None if selected_user == "全部用户" else selected_user

        records = db.get_training_records(limit=200, username=filter_username)
        if records:
            import pandas as pd
            df_rec = pd.DataFrame(records, columns=["ID", "日期", "动作类型", "次数", "平均分", "时长(秒)", "备注"])
            # 清洗脏数据
            for col in df_rec.columns:
                df_rec[col] = df_rec[col].apply(
                    lambda x: str(x, errors="replace") if isinstance(x, bytes) else x)
            df_rec["次数"] = pd.to_numeric(df_rec["次数"], errors="coerce").fillna(0).astype(int)
            df_rec["平均分"] = pd.to_numeric(df_rec["平均分"], errors="coerce").fillna(0).astype(float)
            df_rec["时长(秒)"] = pd.to_numeric(df_rec["时长(秒)"], errors="coerce").fillna(0).astype(float)
            st.dataframe(df_rec, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### 🗑️ 删除记录")
            del_id = st.number_input("输入要删除的记录 ID", min_value=1, step=1)
            if st.button("删除该记录"):
                db.delete_record(int(del_id))
                st.toast(f"✅ 记录 {del_id} 已删除")
                st.rerun()

            st.divider()
            if st.button("⚠️ 清空全部训练记录", type="secondary"):
                st.session_state["admin_confirm_clear"] = True

            if st.session_state.get("admin_confirm_clear"):
                st.error("确定要清空所有训练记录吗？此操作不可撤销！")
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("✅ 确认清空", type="primary"):
                        for r in records:
                            db.delete_record(r[0])
                        st.session_state["admin_confirm_clear"] = False
                        st.toast("✅ 已清空")
                        st.rerun()
                with cc2:
                    if st.button("❌ 取消"):
                        st.session_state["admin_confirm_clear"] = False
                        st.rerun()
        else:
            st.info("暂无训练记录")

    # ===== 数据库概览 =====
    with admin_tab3:
        st.markdown("#### 🗄️ 数据库表结构")

        tables_info = [
            ("users", "用户账号表", [
                ("user_id", "INTEGER PK", "用户ID"),
                ("username", "TEXT UNIQUE", "用户名"),
                ("password_hash", "TEXT", "密码哈希"),
                ("role", "TEXT", "角色 (admin/user)"),
                ("created_at", "TEXT", "注册时间"),
            ]),
            ("user_profile", "用户信息表", [
                ("user_id", "INTEGER PK/FK", "用户ID，对应 users.user_id"),
                ("username", "TEXT UNIQUE", "用户名冗余字段"),
                ("nickname", "TEXT", "昵称"),
                ("gender", "TEXT", "性别"),
                ("age", "INTEGER", "年龄"),
                ("height_cm", "REAL", "身高(cm)"),
                ("weight_kg", "REAL", "体重(kg)"),
                ("fitness_goal", "TEXT", "健身目标"),
                ("created_at", "TEXT", "创建时间"),
            ]),
            ("training_records", "训练记录表", [
                ("record_id", "INTEGER PK", "记录ID"),
                ("user_id", "INTEGER", "所属用户ID"),
                ("train_time", "TEXT", "训练时间"),
                ("action_type", "TEXT", "动作类型"),
                ("repetitions", "INTEGER", "完成次数"),
                ("avg_score", "REAL", "平均评分"),
                ("duration_sec", "REAL", "时长(秒)"),
                ("remark", "TEXT", "备注"),
                ("username", "TEXT", "所属用户"),
            ]),
            ("action_rules", "动作规则表", [
                ("rule_id", "INTEGER PK", "规则ID"),
                ("action_type", "TEXT", "动作类型"),
                ("rule_name", "TEXT", "规则名称"),
                ("rule_value", "REAL", "阈值"),
                ("prompt_text", "TEXT", "提示文本"),
            ]),
            ("action_guides", "动作指南表", [
                ("guide_id", "INTEGER PK", "指南ID"),
                ("action_type", "TEXT UNIQUE", "动作类型"),
                ("description", "TEXT", "动作描述"),
                ("key_points", "TEXT", "动作要领"),
                ("common_mistakes", "TEXT", "常见错误"),
                ("target_muscles", "TEXT", "目标肌群"),
                ("difficulty", "TEXT", "难度等级"),
                ("calories_per_rep", "REAL", "每次热量消耗"),
            ]),
            ("training_plans", "训练计划表", [
                ("plan_id", "INTEGER PK", "计划ID"),
                ("user_id", "INTEGER", "所属用户ID"),
                ("username", "TEXT", "所属用户"),
                ("action_type", "TEXT", "动作类型"),
                ("target_reps", "INTEGER", "目标次数"),
                ("target_score", "REAL", "目标评分"),
                ("plan_date", "TEXT", "计划日期"),
                ("is_completed", "INTEGER", "是否完成(0/1)"),
                ("actual_reps", "INTEGER", "实际次数"),
                ("actual_score", "REAL", "实际评分"),
            ]),
            ("ai_settings", "AI 接入配置表", [
                ("settings_id", "INTEGER PK", "固定主键(1)"),
                ("enabled", "INTEGER", "是否启用 AI 功能(0/1)"),
                ("provider_name", "TEXT", "AI 服务商名称"),
                ("base_url", "TEXT", "模型接口 Base URL"),
                ("api_key", "TEXT", "API Key"),
                ("model_name", "TEXT", "当前使用模型名"),
                ("system_prompt", "TEXT", "补充系统提示词"),
                ("temperature", "REAL", "采样温度"),
                ("timeout_sec", "INTEGER", "接口超时时间(秒)"),
                ("updated_at", "TEXT", "最后更新时间"),
            ]),
            ("ai_report_history", "AI 训练报告历史表", [
                ("report_id", "INTEGER PK", "报告ID"),
                ("user_id", "INTEGER", "所属用户ID"),
                ("username", "TEXT", "所属用户"),
                ("report_time", "TEXT", "报告生成时间"),
                ("provider_name", "TEXT", "生成报告的服务商"),
                ("model_name", "TEXT", "生成报告的模型"),
                ("record_count", "INTEGER", "分析所用训练记录数"),
                ("source_summary", "TEXT", "结构化训练摘要(JSON)"),
                ("report_json", "TEXT", "固定格式报告(JSON)"),
                ("raw_response", "TEXT", "模型原始回复"),
            ]),
            ("ai_chat_sessions", "AI 私教聊天会话表", [
                ("session_id", "INTEGER PK", "聊天会话ID"),
                ("report_id", "INTEGER", "关联报告ID"),
                ("user_id", "INTEGER", "所属用户ID"),
                ("username", "TEXT", "所属用户"),
                ("session_title", "TEXT", "聊天标题"),
                ("created_at", "TEXT", "创建时间"),
                ("updated_at", "TEXT", "最后活跃时间"),
            ]),
            ("ai_report_chat_history", "AI 私教聊天记录表", [
                ("chat_id", "INTEGER PK", "消息ID"),
                ("session_id", "INTEGER", "所属聊天会话ID"),
                ("report_id", "INTEGER", "关联报告ID"),
                ("user_id", "INTEGER", "所属用户ID"),
                ("username", "TEXT", "所属用户"),
                ("role", "TEXT", "消息角色(user/assistant)"),
                ("content", "TEXT", "消息内容"),
                ("created_at", "TEXT", "消息时间"),
            ]),
        ]

        for tbl_name, tbl_desc, columns in tables_info:
            with st.expander(f"📄 {tbl_name} — {tbl_desc}", expanded=False):
                import pandas as pd
                df_schema = pd.DataFrame(columns, columns=["字段名", "类型", "说明"])
                st.dataframe(df_schema, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### 📊 数据统计")
        stats = db.get_statistics()
        s1, s2, s3 = st.columns(3)
        s1.metric("用户数", f"{len(db.get_all_users())}")
        s2.metric("训练记录数", f"{stats['total_sessions']}")
        s3.metric("动作规则数", f"{len(db.get_action_rules())}")


# ==================== 主入口 ====================
def main():
    init_session_state()

    if not st.session_state.get("logged_in"):
        render_login_page()
        return

    role = st.session_state.get("user_role", "user")
    with st.sidebar:
        st.markdown(f"**👤 {st.session_state['username']}**　"
                    f"{'🔑 管理员' if role == 'admin' else '👤 普通用户'}")
        if st.button("🚪 退出登录", use_container_width=True):
            for key in ["logged_in", "username", "user_role", "user_id"]:
                st.session_state[key] = False if key == "logged_in" else ""
            st.rerun()
        st.markdown("---")

        if role == "admin":
            page = st.radio(
                "📌 管理后台",
                ["🔧 系统管理", "🤖 AI 设置"],
                index=0,
            )
        else:
            page = st.radio(
                "📌 页面导航",
                ["🏠 训练主页", "📊 历史记录", "📈 数据可视化", "🤖 AI 专属私教",
                 "📖 动作指南", "📋 训练计划", "👤 个人中心"],
                index=0,
            )

    if role == "admin":
        if page == "🔧 系统管理":
            render_admin_page()
        else:
            render_admin_ai_page()
    elif page == "🏠 训练主页":
        render_main_page()
    elif page == "📊 历史记录":
        render_history_page()
    elif page == "📈 数据可视化":
        render_visualization_page()
    elif page == "🤖 AI 专属私教":
        render_ai_report_page()
    elif page == "📖 动作指南":
        render_guide_page()
    elif page == "📋 训练计划":
        render_plan_page()
    elif page == "👤 个人中心":
        render_profile_page()


if __name__ == "__main__":
    main()
