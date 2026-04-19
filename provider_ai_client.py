# -*- coding: utf-8 -*-
"""
多服务商 AI 客户端
"""
import json
from urllib import error, parse, request


PROVIDER_OPTIONS = [
    {
        "key": "custom_openai",
        "label": "自定义（兼容 OpenAI）",
        "default_base_url": "",
        "auth_hint": "Bearer Token",
    },
    {
        "key": "google_ai_studio",
        "label": "Google AI Studio",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "auth_hint": "x-goog-api-key",
    },
    {
        "key": "openai",
        "label": "OpenAI",
        "default_base_url": "https://api.openai.com/v1",
        "auth_hint": "Bearer Token",
    },
    {
        "key": "claude",
        "label": "Claude",
        "default_base_url": "https://api.anthropic.com",
        "auth_hint": "x-api-key",
    },
]

PROVIDER_BY_KEY = {item["key"]: item for item in PROVIDER_OPTIONS}
PROVIDER_BY_LABEL = {item["label"]: item for item in PROVIDER_OPTIONS}


class AIClientError(Exception):
    """AI 客户端异常"""


class AIConfigError(AIClientError):
    """AI 配置异常"""


class AIResponseError(AIClientError):
    """AI 响应异常"""


def get_provider_option(provider_key=None, provider_name=None):
    """根据 key 或名称获取 provider 配置"""
    if provider_key and provider_key in PROVIDER_BY_KEY:
        return PROVIDER_BY_KEY[provider_key]
    if provider_name and provider_name in PROVIDER_BY_LABEL:
        return PROVIDER_BY_LABEL[provider_name]
    if provider_name:
        lowered = provider_name.lower()
        if "compatible" in lowered or "兼容" in provider_name:
            return PROVIDER_BY_KEY["custom_openai"]
        for item in PROVIDER_OPTIONS:
            if lowered == item["label"].lower():
                return item
        if "google" in lowered or "gemini" in lowered:
            return PROVIDER_BY_KEY["google_ai_studio"]
        if "claude" in lowered or "anthropic" in lowered:
            return PROVIDER_BY_KEY["claude"]
        if "openai" in lowered and "custom" not in lowered and "兼容" not in provider_name:
            return PROVIDER_BY_KEY["openai"]
    return PROVIDER_BY_KEY["custom_openai"]


def provider_label(provider_key):
    return get_provider_option(provider_key=provider_key)["label"]


def provider_default_base_url(provider_key):
    return get_provider_option(provider_key=provider_key)["default_base_url"]


def _decode_json_response(raw_text):
    try:
        return json.loads(raw_text)
    except Exception as exc:
        raise AIResponseError(f"接口返回了非 JSON 数据: {raw_text[:300]}") from exc


def _http_request(url, method="GET", headers=None, payload=None, timeout_sec=60):
    headers = headers or {}
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AIResponseError(f"接口返回 HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise AIResponseError(f"接口连接失败: {exc.reason}") from exc
    except Exception as exc:
        raise AIResponseError(f"接口调用失败: {exc}") from exc
    return _decode_json_response(raw)


class MultiProviderAIClient:
    """按服务商分发的统一 AI 客户端"""

    def __init__(self, settings):
        self.settings = settings or {}
        self.enabled = bool(self.settings.get("enabled", 0))
        provider = get_provider_option(
            provider_key=self.settings.get("provider_key"),
            provider_name=self.settings.get("provider_name"),
        )
        self.provider_key = provider["key"]
        self.provider_name = provider["label"]
        self.base_url = (self.settings.get("base_url") or provider["default_base_url"] or "").strip()
        self.api_key = (self.settings.get("api_key") or "").strip()
        self.model_name = (self.settings.get("model_name") or "").strip()
        self.temperature = float(self.settings.get("temperature", 0.2) or 0.2)
        self.timeout_sec = int(self.settings.get("timeout_sec", 60) or 60)

    def validate(self, require_model=True):
        if not self.enabled:
            raise AIConfigError("AI 报告功能尚未启用，请联系管理员开启。")
        if not self.api_key:
            raise AIConfigError("AI API Key 未配置。")
        if not self.base_url:
            raise AIConfigError("AI Base URL 未配置。")
        if require_model and not self.model_name:
            raise AIConfigError("AI 模型名称未配置。")

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

    def _normalize_openai_base(self):
        base = self.base_url.rstrip("/")
        if not base:
            return "https://api.openai.com/v1" if self.provider_key == "openai" else ""
        if base.endswith("/chat/completions"):
            return base[:-17]
        if base.endswith("/models"):
            return base[:-7]
        return base

    def _build_openai_url(self, path):
        base = self._normalize_openai_base().rstrip("/")
        if base.endswith("/v1"):
            return f"{base}{path}"
        return f"{base}/v1{path}"

    def _normalize_google_base(self):
        base = self.base_url.rstrip("/")
        return base or "https://generativelanguage.googleapis.com/v1beta"

    def _normalize_claude_base(self):
        base = self.base_url.rstrip("/")
        return base or "https://api.anthropic.com"

    def create_chat_completion(self, system_prompt, user_prompt, expect_json=False):
        self.validate(require_model=True)
        if self.provider_key in ("custom_openai", "openai"):
            return self._create_openai_chat_completion(system_prompt, user_prompt, expect_json=expect_json)
        if self.provider_key == "google_ai_studio":
            return self._create_google_chat_completion(system_prompt, user_prompt)
        if self.provider_key == "claude":
            return self._create_claude_chat_completion(system_prompt, user_prompt)
        raise AIConfigError(f"不支持的服务商类型: {self.provider_key}")

    def fetch_models(self):
        self.validate(require_model=False)
        if self.provider_key in ("custom_openai", "openai"):
            return self._fetch_openai_models()
        if self.provider_key == "google_ai_studio":
            return self._fetch_google_models()
        if self.provider_key == "claude":
            return self._fetch_claude_models()
        raise AIConfigError(f"不支持的服务商类型: {self.provider_key}")

    def _create_openai_chat_completion(self, system_prompt, user_prompt, expect_json=False):
        url = self._build_openai_url("/chat/completions")
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        if expect_json:
            payload["response_format"] = {"type": "json_object"}
        data = _http_request(
            url,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            payload=payload,
            timeout_sec=self.timeout_sec,
        )
        choices = data.get("choices") or []
        if not choices:
            raise AIResponseError("模型响应中缺少 choices。")
        message = choices[0].get("message") or {}
        content = self._extract_text_content(message.get("content"))
        if not content:
            raise AIResponseError("模型响应内容为空。")
        return content, data

    def _fetch_openai_models(self):
        url = self._build_openai_url("/models")
        data = _http_request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout_sec=self.timeout_sec,
        )
        models = []
        for item in data.get("data", []):
            model_id = (item or {}).get("id", "").strip()
            if model_id:
                models.append(model_id)
        return sorted(set(models))

    def _create_google_chat_completion(self, system_prompt, user_prompt):
        base = self._normalize_google_base()
        model_path = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"
        url = f"{base}/{model_path}:generateContent"
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
            },
        }
        data = _http_request(
            url,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            payload=payload,
            timeout_sec=self.timeout_sec,
        )
        candidates = data.get("candidates") or []
        if not candidates:
            raise AIResponseError("Gemini 响应中缺少 candidates。")
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
        if not text_parts:
            raise AIResponseError("Gemini 响应内容为空。")
        return "\n".join(text_parts), data

    def _fetch_google_models(self):
        base = self._normalize_google_base()
        url = f"{base}/models"
        data = _http_request(
            url,
            headers={"x-goog-api-key": self.api_key},
            timeout_sec=self.timeout_sec,
        )
        models = []
        for item in data.get("models", []):
            methods = item.get("supportedGenerationMethods") or []
            if methods and "generateContent" not in methods:
                continue
            name = (item.get("name") or "").strip()
            if name.startswith("models/"):
                name = name[len("models/"):]
            if name:
                models.append(name)
        return sorted(set(models))

    def _create_claude_chat_completion(self, system_prompt, user_prompt):
        base = self._normalize_claude_base()
        url = f"{base}/v1/messages"
        payload = {
            "model": self.model_name,
            "max_tokens": 2048,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
        }
        data = _http_request(
            url,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            payload=payload,
            timeout_sec=self.timeout_sec,
        )
        content = data.get("content") or []
        text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("text")]
        if not text_parts:
            raise AIResponseError("Claude 响应内容为空。")
        return "\n".join(text_parts), data

    def _fetch_claude_models(self):
        base = self._normalize_claude_base()
        url = f"{base}/v1/models"
        data = _http_request(
            url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout_sec=self.timeout_sec,
        )
        models = []
        for item in data.get("data", []):
            model_id = (item.get("id") or "").strip()
            if model_id:
                models.append(model_id)
        return sorted(set(models))


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
