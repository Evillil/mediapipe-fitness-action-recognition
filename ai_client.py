# -*- coding: utf-8 -*-
"""
OpenAI 兼容接口客户端
"""
import json
from urllib import error, request


class AIClientError(Exception):
    """AI 客户端异常"""


class AIConfigError(AIClientError):
    """AI 配置异常"""


class AIResponseError(AIClientError):
    """AI 响应异常"""


class OpenAICompatibleClient:
    """调用 OpenAI 兼容的 Chat Completions 接口"""

    def __init__(self, settings):
        self.settings = settings or {}
        self.enabled = bool(self.settings.get("enabled", 0))
        self.provider_name = (self.settings.get("provider_name") or "OpenAI Compatible").strip()
        self.base_url = (self.settings.get("base_url") or "").strip()
        self.api_key = (self.settings.get("api_key") or "").strip()
        self.model_name = (self.settings.get("model_name") or "").strip()
        self.temperature = float(self.settings.get("temperature", 0.2) or 0.2)
        self.timeout_sec = int(self.settings.get("timeout_sec", 60) or 60)

    def validate(self):
        if not self.enabled:
            raise AIConfigError("AI 报告功能尚未启用，请联系管理员开启。")
        if not self.base_url:
            raise AIConfigError("AI Base URL 未配置。")
        if not self.api_key:
            raise AIConfigError("AI API Key 未配置。")
        if not self.model_name:
            raise AIConfigError("AI 模型名称未配置。")

    def _build_endpoint(self):
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/chat/completions"

    @staticmethod
    def _extract_text_content(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(text)
            return "\n".join(parts)
        return str(content)

    def create_chat_completion(self, system_prompt, user_prompt):
        self.validate()

        endpoint = self._build_endpoint()
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = request.Request(endpoint, data=body, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AIResponseError(f"AI 接口返回 HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise AIResponseError(f"AI 接口连接失败: {exc.reason}") from exc
        except Exception as exc:
            raise AIResponseError(f"AI 调用失败: {exc}") from exc

        try:
            data = json.loads(raw)
        except Exception as exc:
            raise AIResponseError(f"AI 返回了非 JSON 数据: {raw[:300]}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise AIResponseError(f"AI 返回中缺少 choices: {raw[:300]}")

        message = choices[0].get("message") or {}
        content = self._extract_text_content(message.get("content"))
        if not content:
            raise AIResponseError(f"AI 返回内容为空: {raw[:300]}")

        return content, data


def parse_json_response(text):
    """从模型文本响应中提取 JSON"""
    if not text:
        raise AIResponseError("AI 返回内容为空。")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise AIResponseError("AI 返回内容中未找到有效 JSON。")
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception as exc:
            raise AIResponseError(f"AI 返回 JSON 解析失败: {cleaned[:500]}") from exc
