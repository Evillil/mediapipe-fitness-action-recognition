# -*- coding: utf-8 -*-
"""
AI 训练报告服务
"""
import json
import os
from io import BytesIO
from datetime import datetime, timedelta

from PIL import Image, ImageDraw, ImageFont

from provider_ai_client import AIConfigError, AIResponseError, MultiProviderAIClient, parse_json_response
from ai_store import AIStore


class AIReportError(Exception):
    """AI 报告生成异常"""


class AICoachError(Exception):
    """AI 私教异常"""


DEFAULT_SYSTEM_PROMPT = """
你是一名健身训练数据分析助手。你的任务是基于系统提供的用户资料、训练记录、训练计划和统计结果，
输出一份客观、克制、实用的中文训练分析报告。

规则：
1. 只能依据给定数据分析，不要虚构用户行为和身体情况。
2. 如果数据不足，要明确指出“不足以判断”。
3. 不要做医学诊断，不要承诺治疗效果。
4. 建议必须具体、可执行，尽量结合用户目标和计划执行情况。
5. 输出必须是合法 JSON，且只能输出 JSON，不要加解释文字，不要用 Markdown 代码块。

JSON 字段固定为：
{
  "report_title": "字符串",
  "summary": "字符串",
  "user_profile": ["字符串"],
  "training_overview": ["字符串"],
  "goal_alignment": ["字符串"],
  "plan_execution": ["字符串"],
  "key_problems": ["字符串"],
  "training_suggestions": ["字符串"],
  "next_7_days_actions": ["字符串"],
  "risk_alerts": ["字符串"]
}
""".strip()


DEFAULT_COACH_CHAT_PROMPT = """
你是一名中文健身私教助手。你只能基于系统给出的训练报告、结构化统计摘要和历史对话回答问题。

规则：
1. 不要编造报告中没有的数据、病症或身体结论。
2. 如果报告不足以支持判断，要明确说明“当前报告信息不足”。
3. 不要做医学诊断，不要给出危险训练指令。
4. 回答要直接、具体、实用，优先给出结论、依据和建议。
5. 直接输出中文正文，不要输出 JSON，不要输出 Markdown 代码块。
""".strip()


PDF_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _bmi_status(bmi_value):
    if bmi_value <= 0:
        return "未知"
    if bmi_value < 18.5:
        return "偏瘦"
    if bmi_value < 24:
        return "正常"
    if bmi_value < 28:
        return "偏胖"
    return "肥胖"


def _to_record_dict(record):
    return {
        "record_id": _safe_int(record[0]),
        "train_time": record[1],
        "action_type": record[2],
        "repetitions": _safe_int(record[3]),
        "avg_score": round(_safe_float(record[4]), 1),
        "duration_sec": round(_safe_float(record[5]), 1),
        "remark": record[6] or "",
    }


def _to_plan_dict(plan):
    return {
        "plan_id": _safe_int(plan[0]),
        "username": plan[1],
        "action_type": plan[2],
        "target_reps": _safe_int(plan[3]),
        "target_score": round(_safe_float(plan[4]), 1),
        "plan_date": plan[5],
        "is_completed": bool(_safe_int(plan[6])),
        "actual_reps": _safe_int(plan[7]),
        "actual_score": round(_safe_float(plan[8]), 1),
    }


def build_analysis_payload(db, username=None, user_id=None, max_records=50):
    """汇总用户分析数据"""
    resolved_username = username
    if user_id is not None and not resolved_username:
        user_row = db.get_user_by_id(user_id)
        if user_row:
            resolved_username = user_row[1]

    profile = db.get_user_profile(username=resolved_username, user_id=user_id) or {}
    records = db.get_training_records(limit=max_records, username=resolved_username, user_id=user_id)
    if not records:
        raise AIReportError("当前用户还没有训练记录，无法生成 AI 报告。")

    plans = db.get_plans(username=resolved_username, user_id=user_id)
    unique_plan_dates = sorted({plan[5] for plan in plans})
    for plan_date in unique_plan_dates:
        db.sync_plans_with_records(username=resolved_username, user_id=user_id, plan_date=plan_date)
    plans = db.get_plans(username=resolved_username, user_id=user_id)

    stats = db.get_statistics(username=resolved_username, user_id=user_id)
    record_dicts = [_to_record_dict(record) for record in records]
    plan_dicts = [_to_plan_dict(plan) for plan in plans]

    height_cm = _safe_float(profile.get("height_cm"), 0.0)
    weight_kg = _safe_float(profile.get("weight_kg"), 0.0)
    bmi_value = round(weight_kg / ((height_cm / 100.0) ** 2), 1) if height_cm > 0 and weight_kg > 0 else 0.0

    now = datetime.now()
    week_cutoff = now - timedelta(days=7)
    month_cutoff = now - timedelta(days=30)
    records_last_7d = []
    records_last_30d = []
    latest_train_time = ""

    for item in record_dicts:
        train_time = item.get("train_time", "")
        try:
            dt = datetime.strptime(train_time, "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        if not latest_train_time:
            latest_train_time = train_time
        if dt >= week_cutoff:
            records_last_7d.append(item)
        if dt >= month_cutoff:
            records_last_30d.append(item)

    per_action = []
    best_action = ""
    most_reps_action = ""
    best_score = -1.0
    max_reps = -1
    for action_type, sessions, total_reps, avg_score in stats.get("per_action", []):
        action_item = {
            "action_type": action_type,
            "sessions": _safe_int(sessions),
            "total_reps": _safe_int(total_reps),
            "avg_score": round(_safe_float(avg_score), 1),
        }
        per_action.append(action_item)
        if action_item["avg_score"] > best_score:
            best_score = action_item["avg_score"]
            best_action = action_type
        if action_item["total_reps"] > max_reps:
            max_reps = action_item["total_reps"]
            most_reps_action = action_type

    completed_plans = sum(1 for plan in plan_dicts if plan["is_completed"])
    today_str = now.strftime("%Y-%m-%d")
    today_plans = [plan for plan in plan_dicts if plan["plan_date"] == today_str]
    future_plans = [plan for plan in plan_dicts if plan["plan_date"] > today_str][:5]

    payload = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": _safe_int(user_id, 0),
        "username": resolved_username or "",
        "profile": {
            "nickname": profile.get("nickname", ""),
            "gender": profile.get("gender", ""),
            "age": _safe_int(profile.get("age"), 0),
            "height_cm": round(height_cm, 1),
            "weight_kg": round(weight_kg, 1),
            "fitness_goal": profile.get("fitness_goal", ""),
            "bmi": bmi_value,
            "bmi_status": _bmi_status(bmi_value),
        },
        "training_summary": {
            "total_sessions": _safe_int(stats.get("total_sessions"), 0),
            "total_reps": _safe_int(stats.get("total_reps"), 0),
            "total_duration_min": round(_safe_float(stats.get("total_duration"), 0.0) / 60.0, 1),
            "avg_score": round(_safe_float(stats.get("avg_score"), 0.0), 1),
            "latest_train_time": latest_train_time,
            "recent_record_count": len(record_dicts),
            "records_last_7d": {
                "sessions": len(records_last_7d),
                "total_reps": sum(item["repetitions"] for item in records_last_7d),
                "avg_score": round(
                    sum(item["avg_score"] for item in records_last_7d) / len(records_last_7d), 1
                ) if records_last_7d else 0.0,
            },
            "records_last_30d": {
                "sessions": len(records_last_30d),
                "total_reps": sum(item["repetitions"] for item in records_last_30d),
            },
            "best_action_by_score": best_action,
            "most_practiced_action": most_reps_action,
            "per_action": per_action,
            "recent_records": record_dicts[:min(len(record_dicts), 20)],
        },
        "plan_summary": {
            "plan_count": len(plan_dicts),
            "completed_plan_count": completed_plans,
            "completion_rate": round((completed_plans / len(plan_dicts) * 100.0), 1) if plan_dicts else 0.0,
            "today_plans": today_plans,
            "future_plans": future_plans,
            "recent_plans": plan_dicts[:min(len(plan_dicts), 10)],
        },
    }
    return payload


def build_report_prompt(payload):
    return (
        "请基于下面的 JSON 数据输出固定格式报告。\n"
        "如果用户没有训练计划或个人资料，要在对应分析里直接说明。\n"
        "不要输出 Markdown。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _normalize_string_list(value):
    if isinstance(value, list):
        result = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def normalize_report(report_json, username):
    """规范化模型输出"""
    normalized = {
        "report_title": str(report_json.get("report_title") or f"{username} 的训练分析报告").strip(),
        "summary": str(report_json.get("summary") or "暂无摘要。").strip(),
        "user_profile": _normalize_string_list(report_json.get("user_profile")),
        "training_overview": _normalize_string_list(report_json.get("training_overview")),
        "goal_alignment": _normalize_string_list(report_json.get("goal_alignment")),
        "plan_execution": _normalize_string_list(report_json.get("plan_execution")),
        "key_problems": _normalize_string_list(report_json.get("key_problems")),
        "training_suggestions": _normalize_string_list(report_json.get("training_suggestions")),
        "next_7_days_actions": _normalize_string_list(report_json.get("next_7_days_actions")),
        "risk_alerts": _normalize_string_list(report_json.get("risk_alerts")),
    }
    return normalized


def build_report_export_blocks(report_detail):
    """构建导出用的报告文本块"""
    report = report_detail.get("report_json", {}) if report_detail else {}
    source = report_detail.get("source_summary", {}) if report_detail else {}
    training_summary = source.get("training_summary", {})
    plan_summary = source.get("plan_summary", {})
    profile = source.get("profile", {})

    blocks = [
        ("title", report.get("report_title", "AI 专属私教训练报告")),
        ("meta", f"生成时间：{report_detail.get('report_time', '')}"),
        ("meta", f"模型：{report_detail.get('provider_name', '')} / {report_detail.get('model_name', '')}"),
        ("blank", ""),
        ("section", "核心指标"),
        ("bullet", f"分析记录数：{training_summary.get('recent_record_count', 0)} 条"),
        ("bullet", f"累计训练次数：{training_summary.get('total_sessions', 0)} 次"),
        ("bullet", f"累计动作数：{training_summary.get('total_reps', 0)} 个"),
        ("bullet", f"计划完成率：{plan_summary.get('completion_rate', 0):.1f}%"),
    ]

    summary = str(report.get("summary") or "").strip()
    if summary:
        blocks.extend([
            ("blank", ""),
            ("section", "总结"),
            ("body", summary),
        ])

    if profile:
        blocks.extend([
            ("blank", ""),
            ("section", "基础信息"),
            ("bullet", f"昵称：{profile.get('nickname', '未填写') or '未填写'}"),
            ("bullet", f"性别：{profile.get('gender', '未填写') or '未填写'}"),
            ("bullet", f"年龄：{profile.get('age', 0) or '未填写'}"),
            ("bullet", f"身高：{profile.get('height_cm', 0) or '未填写'} cm"),
            ("bullet", f"体重：{profile.get('weight_kg', 0) or '未填写'} kg"),
            ("bullet", f"健身目标：{profile.get('fitness_goal', '未填写') or '未填写'}"),
            ("bullet", f"BMI：{profile.get('bmi', 0)} ({profile.get('bmi_status', '未知')})"),
        ])

    sections = [
        ("用户概况", report.get("user_profile", []), "未生成用户概况分析。"),
        ("训练总览", report.get("training_overview", []), "未生成训练总览。"),
        ("目标匹配度分析", report.get("goal_alignment", []), "未生成目标匹配度分析。"),
        ("计划执行情况", report.get("plan_execution", []), "当前没有计划执行分析。"),
        ("主要问题", report.get("key_problems", []), "当前没有明确识别出主要问题。"),
        ("训练建议", report.get("training_suggestions", []), "未生成训练建议。"),
        ("未来 7 天行动清单", report.get("next_7_days_actions", []), "未生成行动清单。"),
        ("风险与注意事项", report.get("risk_alerts", []), "当前没有额外风险提示。"),
    ]

    for title, items, empty_text in sections:
        blocks.append(("blank", ""))
        blocks.append(("section", title))
        if items:
            for item in items:
                blocks.append(("bullet", str(item).strip()))
        else:
            blocks.append(("body", empty_text))
    return blocks


def _get_pdf_font_path():
    for font_path in PDF_FONT_CANDIDATES:
        if os.path.exists(font_path):
            return font_path
    raise AICoachError("当前系统未找到可用的中文字体，暂时无法导出 PDF。")


def _measure_text(draw, text, font):
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _wrap_pdf_text(draw, text, font, max_width):
    text = str(text or "")
    if not text:
        return [""]

    first_prefix = ""
    next_prefix = ""
    body = text
    if text.startswith("• "):
        first_prefix = "• "
        next_prefix = "  "
        body = text[2:]
    elif text.startswith("- "):
        first_prefix = "- "
        next_prefix = "  "
        body = text[2:]

    wrapped_lines = []
    paragraphs = body.splitlines() or [body]
    for paragraph in paragraphs:
        if not paragraph:
            wrapped_lines.append("")
            continue

        prefix = first_prefix
        current = prefix
        for ch in paragraph:
            candidate = current + ch
            if _measure_text(draw, candidate, font) <= max_width or current == prefix:
                current = candidate
            else:
                wrapped_lines.append(current)
                prefix = next_prefix
                current = prefix + ch
        if current:
            wrapped_lines.append(current)
        first_prefix = ""
        next_prefix = ""
    return wrapped_lines or [text]


def build_report_pdf_bytes(report_detail):
    """生成报告 PDF 二进制内容"""
    font_path = _get_pdf_font_path()
    page_width = 1654
    page_height = 2339
    margin_x = 120
    margin_y = 120
    text_width = page_width - margin_x * 2

    font_specs = {
        "title": (ImageFont.truetype(font_path, 48), 18, 26),
        "section": (ImageFont.truetype(font_path, 30), 12, 18),
        "meta": (ImageFont.truetype(font_path, 22), 8, 8),
        "body": (ImageFont.truetype(font_path, 24), 10, 12),
        "bullet": (ImageFont.truetype(font_path, 24), 10, 8),
    }

    def new_page():
        image = Image.new("RGB", (page_width, page_height), "white")
        return image, ImageDraw.Draw(image)

    blocks = build_report_export_blocks(report_detail)
    pages = []
    page, draw = new_page()
    y = margin_y

    def ensure_space(required_height):
        nonlocal page, draw, y
        if y + required_height <= page_height - margin_y:
            return
        pages.append(page)
        page, draw = new_page()
        y = margin_y

    for kind, text in blocks:
        if kind == "blank":
            y += 12
            continue

        font, line_gap, block_gap = font_specs.get(kind, font_specs["body"])
        line_text = f"• {text}" if kind == "bullet" else text
        wrapped = _wrap_pdf_text(draw, line_text, font, text_width)
        line_height = font.size + line_gap
        block_height = len(wrapped) * line_height + block_gap
        ensure_space(block_height)

        fill = (20, 20, 20)
        if kind == "title":
            fill = (15, 35, 75)
        elif kind == "section":
            fill = (40, 40, 40)
        elif kind == "meta":
            fill = (90, 90, 90)

        for line in wrapped:
            draw.text((margin_x, y), line, fill=fill, font=font)
            y += line_height
        y += block_gap

    pages.append(page)
    output = BytesIO()
    rgb_pages = [item.convert("RGB") for item in pages]
    rgb_pages[0].save(output, format="PDF", save_all=True, append_images=rgb_pages[1:])
    return output.getvalue()


def _build_chat_history_text(messages, limit=10):
    if not messages:
        return "暂无历史问答。"

    rows = []
    for item in messages[-limit:]:
        role = "用户" if item.get("role") == "user" else "AI私教"
        content = str(item.get("content") or "").strip()
        if content:
            rows.append(f"{role}：{content}")
    return "\n".join(rows) if rows else "暂无历史问答。"


def _clean_chat_answer(text):
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("markdown"):
            cleaned = cleaned[8:].strip()
        elif cleaned.startswith("text"):
            cleaned = cleaned[4:].strip()
    return cleaned.strip()


def generate_ai_coach_reply(report_detail, question, ai_store=None, history_limit=10, session_id=None):
    """基于已生成报告回答追问"""
    question = str(question or "").strip()
    if not question:
        raise AICoachError("请输入要追问的问题。")

    if not report_detail:
        raise AICoachError("未找到对应的训练报告。")

    ai_store = ai_store or AIStore()
    settings = ai_store.get_ai_settings()
    report_id = report_detail.get("report_id")
    username = report_detail.get("username")
    user_id = report_detail.get("user_id")
    history_messages = ai_store.get_report_chat_messages(
        report_id,
        username=username,
        user_id=user_id,
        limit=history_limit,
        session_id=session_id,
    )

    custom_prompt = (settings.get("system_prompt") or "").strip()
    system_prompt = DEFAULT_COACH_CHAT_PROMPT
    if custom_prompt:
        system_prompt = f"{system_prompt}\n\n补充要求：\n{custom_prompt}"

    user_prompt = (
        "以下是当前被追问的训练报告与统计摘要，请仅基于这些信息回答。\n\n"
        f"[训练报告]\n{json.dumps(report_detail.get('report_json', {}), ensure_ascii=False, indent=2)}\n\n"
        f"[统计摘要]\n{json.dumps(report_detail.get('source_summary', {}), ensure_ascii=False, indent=2)}\n\n"
        f"[最近对话]\n{_build_chat_history_text(history_messages, limit=history_limit)}\n\n"
        f"[用户当前问题]\n{question}\n\n"
        "请直接给出中文回答。优先包含：结论、依据、建议。"
    )

    client = MultiProviderAIClient(settings)
    try:
        raw_text, raw_response = client.create_chat_completion(system_prompt, user_prompt, expect_json=False)
    except (AIConfigError, AIResponseError) as exc:
        raise AICoachError(str(exc)) from exc
    except Exception as exc:
        raise AICoachError(f"AI 私教回复失败: {exc}") from exc

    answer = _clean_chat_answer(raw_text)
    if not answer:
        raise AICoachError("AI 私教返回内容为空。")

    return {
        "answer": answer,
        "raw_response_text": raw_text,
        "raw_response_json": raw_response,
        "provider_name": client.provider_name,
        "model_name": client.model_name,
    }


def generate_ai_training_report(db, username=None, user_id=None, max_records=50, ai_store=None):
    """生成 AI 训练分析报告"""
    ai_store = ai_store or AIStore()
    settings = ai_store.get_ai_settings()
    payload = build_analysis_payload(db, username=username, user_id=user_id, max_records=max_records)
    resolved_username = payload.get("username") or username or f"用户{_safe_int(user_id, 0)}"

    custom_prompt = (settings.get("system_prompt") or "").strip()
    system_prompt = DEFAULT_SYSTEM_PROMPT if not custom_prompt else f"{DEFAULT_SYSTEM_PROMPT}\n\n补充要求：\n{custom_prompt}"
    user_prompt = build_report_prompt(payload)

    client = MultiProviderAIClient(settings)
    try:
        raw_text, raw_response = client.create_chat_completion(system_prompt, user_prompt, expect_json=True)
        parsed = parse_json_response(raw_text)
    except (AIConfigError, AIResponseError) as exc:
        raise AIReportError(str(exc)) from exc
    except Exception as exc:
        raise AIReportError(f"AI 报告生成失败: {exc}") from exc

    report = normalize_report(parsed, username=resolved_username)
    return {
        "report": report,
        "source_summary": payload,
        "raw_response_text": raw_text,
        "raw_response_json": raw_response,
        "provider_name": client.provider_name,
        "model_name": client.model_name,
        "user_id": _safe_int(user_id, 0),
        "username": resolved_username,
        "record_count": payload["training_summary"]["recent_record_count"],
    }
