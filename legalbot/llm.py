"""LLM 客户端与 JSON 调用。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib import request, error

from . import config


def load_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        load_env()
        self.config = config or self._from_env()

    def _from_env(self) -> LLMConfig:
        provider = os.getenv("LEGALBOT_LLM_PROVIDER", "deepseek").lower()
        if provider == "deepseek":
            key = os.getenv("DEEPSEEK_API_KEY", "")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
            model = os.getenv("LEGALBOT_LLM_MODEL") or _first_model(os.getenv("DEEPSEEK_MODELS", "deepseek-chat"))
            if not key:
                raise LLMError("DEEPSEEK_API_KEY 未配置，无法使用 LLM 驱动问答。")
            return LLMConfig(provider="deepseek", api_key=key, base_url=base_url, model=model)
        if provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            model = os.getenv("LEGALBOT_LLM_MODEL") or _first_model(os.getenv("OPENAI_MODELS", "gpt-4o"))
            if not key:
                raise LLMError("OPENAI_API_KEY 未配置，无法使用 LLM 驱动问答。")
            return LLMConfig(provider="openai", api_key=key, base_url=base_url, model=model)
        if provider.startswith("custom"):
            suffix = provider.replace("custom", "") or "1"
            key = os.getenv(f"CUSTOM_API_KEY_{suffix}", "")
            base_url = os.getenv(f"CUSTOM_API_BASE_URL_{suffix}", "").rstrip("/")
            model = os.getenv("LEGALBOT_LLM_MODEL") or _first_model(os.getenv(f"CUSTOM_API_MODELS_{suffix}", ""))
            if not key or not base_url or not model:
                raise LLMError(f"CUSTOM_API_{suffix} 配置不完整，无法使用 LLM 驱动问答。")
            return LLMConfig(provider=provider, api_key=key, base_url=base_url, model=model)
        raise LLMError(f"不支持的 LEGALBOT_LLM_PROVIDER: {provider}")

    def chat(self, messages: list[dict], temperature: float = 0.0,
             json_mode: bool = False) -> str:
        url = f"{self.config.base_url}/v1/chat/completions"
        if self.config.base_url.endswith("/v1"):
            url = f"{self.config.base_url}/chat/completions"
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            # DeepSeek JSON mode 需要 system message 中提示输出 JSON
            for m in messages:
                if m["role"] == "system":
                    m["content"] += "\n你必须输出合法的JSON格式。"
                    break
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        last_exc = None
        for attempt in range(config.MAX_RETRIES):
            try:
                with request.urlopen(req, timeout=getattr(config, 'LLM_TIMEOUT', 60)) as resp:
                    raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                return parsed["choices"][0]["message"]["content"]
            except error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503) and attempt < config.MAX_RETRIES - 1:
                    import time
                    time.sleep(2 ** attempt)
                    last_exc = exc
                    continue
                body = exc.read().decode("utf-8", errors="ignore")
                # 402 Payment Required：余额耗尽，必须明确提示
                if exc.code == 402:
                    raise LLMError(
                        f"LLM API 余额耗尽（HTTP 402）。请充值后再调用。响应: {body[:200]}"
                    ) from exc
                raise LLMError(f"LLM HTTP {exc.code}: {body[:500]}") from exc
            except (TimeoutError, OSError) as exc:
                if attempt < config.MAX_RETRIES - 1:
                    import time
                    time.sleep(2 ** attempt)
                    last_exc = exc
                    continue
                raise LLMError(f"LLM 调用失败（重试 {attempt + 1} 次）: {exc}") from exc
        raise LLMError(f"LLM 调用失败: {last_exc}") from last_exc

    def json(self, messages: list[dict], temperature: float = 0.0) -> dict:
        """调用 LLM 并解析 JSON。失败时重试 + 兜底返回 {}。

        防御性设计：LLM 偶发返回非法 JSON（如缺逗号），不应让整个 pipeline crash。
        - 重试一次（追加 JSON 格式提醒）
        - 最终失败返回 {}（让调用方用 .get() 拿默认值）
        """
        import time as _time
        last_error: Exception | None = None
        for attempt in range(max(1, config.MAX_RETRIES)):
            try:
                text = self.chat(messages, temperature=temperature, json_mode=True)
                return parse_json_object(text)
            except (json.JSONDecodeError, LLMError) as e:
                last_error = e
                if attempt < max(1, config.MAX_RETRIES) - 1:
                    _time.sleep(2 ** attempt)
                    # 在 system message 追加提醒
                    messages = list(messages)
                    if messages and messages[0].get("role") == "system":
                        messages[0] = {
                            **messages[0],
                            "content": (
                                messages[0]["content"]
                                + "\n\n【格式提醒】你上一次的回复不是合法 JSON。务必："
                                "1) 用双引号包裹所有键和字符串值；"
                                "2) 字段之间用逗号分隔（最后一个字段后不加逗号）；"
                                "3) 完整闭合所有花括号和方括号。"
                            ),
                        }
                    continue
        # 最终失败：返回 {}，让调用方用 .get() 拿默认值
        return {}


def parse_json_object(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    if text.startswith("{"):
        return json.loads(_repair_truncated_json(text))
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise LLMError(f"LLM 未返回 JSON: {text[:300]}")
    return json.loads(_repair_truncated_json(match.group(0)))


def _repair_truncated_json(text: str) -> str:
    """Repair the common case where an LLM returns a JSON object missing closing braces."""
    open_braces = text.count("{")
    close_braces = text.count("}")
    if open_braces > close_braces:
        return text + ("}" * (open_braces - close_braces))
    return text


def _first_model(value: str) -> str:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise LLMError("未配置模型名称。")
    return parts[0]
