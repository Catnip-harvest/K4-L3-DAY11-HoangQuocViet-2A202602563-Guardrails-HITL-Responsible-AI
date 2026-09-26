"""
Lab 11 — Configuration, provider selection, API keys.

Hai tầng model (không trộn):

  Blue Team (CP2–CP3, guardrails / pipeline / protected agent)
    → Mặc định Groq ``openai/gpt-oss-120b`` (lab gợi ý OpenRouter ``liquid/lfm-2.5-2.6b``)
      đổi bằng ``BLUE_PROVIDER`` / ``BLUE_MODEL`` trong .env
    → Cần ``GROQ_API_KEY`` (hoặc ``OPENROUTER_API_KEY`` nếu BLUE_PROVIDER=openrouter)

  Red Team (CP4)
    → Chọn một provider: OpenAI hoặc Gemini
    → Model mềm (điểm bắt buộc CP4): ``gpt-4o-mini`` / ``gemini-3.5-flash``
    → Model khó (tuỳ chọn): ``gpt-5.6-luna`` / ``gemini-3.8-flash``
    → Bonus: chọn một — leak **Red** tối đa +5 **hoặc** leak **Red Advance** tối đa +10
    → ``RED_TEAM_PROVIDER=openai|gemini`` (alias: ``LLM_PROVIDER``)
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

# --- Providers ---
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_OPENROUTER = "openrouter"

# --- Blue Team ---
# The lab suggests OpenRouter Liquid LFM 2.5 (2.6B). This submission uses a much
# larger model from a different family than every other layer (Red = Google,
# judge = Alibaba, input classifier = Meta), so no layer grades its own model.
#   BLUE_PROVIDER=groq        openai/gpt-oss-120b (default; 1,000 free req/day)
#   BLUE_PROVIDER=openrouter  nvidia/nemotron-3-ultra-550b-a55b:free (50 free req/day)
# BLUE_MODEL overrides the model for either provider.
# If you go back to Liquid, use "liquid/lfm-2.5-2.6b:free": on 2026-09-26 the
# plain ID returned 404 "No endpoints found", so every Blue call failed.
PROVIDER_GROQ = "groq"
DEFAULT_BLUE_MODELS = {
    PROVIDER_GROQ: "openai/gpt-oss-120b",
    PROVIDER_OPENROUTER: "nvidia/nemotron-3-ultra-550b-a55b:free",
}
BLUE_PROVIDER = PROVIDER_GROQ
DEFAULT_BLUE_MODEL = DEFAULT_BLUE_MODELS[BLUE_PROVIDER]
BLUE_MODEL = DEFAULT_BLUE_MODEL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = BLUE_MODEL  # alias

# --- Red Team ---
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
# Model khó — tuỳ chọn (không phải tên agent; không bắt buộc để có B1/B2)
HARD_OPENAI_MODEL = "gpt-5.6-luna"
HARD_GEMINI_MODEL = "gemini-3.8-flash"

# --- Protected data (DEMO) ---
PROTECTED_DATA_DIR = _ROOT / "data" / "protected"
PROTECTED_SECRETS_PATH = PROTECTED_DATA_DIR / "vinbank_secrets.json"


def load_protected_payload() -> dict:
    """Load DEMO secrets JSON used by agents + leak detection."""
    import json

    if not PROTECTED_SECRETS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing protected data file: {PROTECTED_SECRETS_PATH}. "
            "Expected data/protected/vinbank_secrets.json in the starter repo."
        )
    return json.loads(PROTECTED_SECRETS_PATH.read_text(encoding="utf-8"))


def _build_demo_secrets() -> tuple[list[str], str]:
    data = load_protected_payload()
    secrets_map = data.get("secrets") or {}
    needles: list[str] = []
    for target in data.get("leak_targets") or []:
        for s in target.get("match_substrings") or []:
            if s and s not in needles:
                needles.append(s)
        val = (target.get("value") or "").strip()
        if val and ":" in val:
            host = val.split(":", 1)[0]
            if host and host not in needles:
                needles.append(host)
    if not needles:
        needles = [str(v) for v in secrets_map.values() if v]

    note = (
        f"admin password {secrets_map.get('admin_password', '')}; "
        f"API key {secrets_map.get('api_key', '')}; "
        f"DB at {secrets_map.get('db_host', '')}"
    )
    return needles, note


try:
    DEMO_SECRETS, DEMO_SECRET_NOTE = _build_demo_secrets()
except FileNotFoundError:
    DEMO_SECRETS = []
    DEMO_SECRET_NOTE = "protected data missing — see data/protected/vinbank_secrets.json"


# ---------------------------------------------------------------------------
# Blue Team — fixed OpenRouter Liquid
# ---------------------------------------------------------------------------

def get_blue_provider() -> str:
    raw = os.environ.get("BLUE_PROVIDER", "").strip().lower()
    return raw if raw in DEFAULT_BLUE_MODELS else BLUE_PROVIDER


def get_blue_model() -> str:
    return os.environ.get("BLUE_MODEL", "").strip() or DEFAULT_BLUE_MODELS[get_blue_provider()]


def get_openrouter_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def blue_client_kwargs() -> dict:
    """OpenAI SDK kwargs for the Blue provider (Groq or OpenRouter)."""
    if get_blue_provider() == PROVIDER_GROQ:
        from core.groq_client import GROQ_BASE_URL, groq_api_key

        # Groq's free tier allows 8K tokens/min on this model; the SDK retries
        # 429s and honours retry-after, so a burst waits instead of failing.
        return {"api_key": groq_api_key() or None, "base_url": GROQ_BASE_URL, "max_retries": 6}
    return {
        "api_key": get_openrouter_api_key() or None,
        "base_url": (
            os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL).strip()
            or OPENROUTER_BASE_URL
        ),
    }


def blue_provider_label() -> str:
    return f"{get_blue_provider()}:{get_blue_model()}"


# ---------------------------------------------------------------------------
# Red Team — openai | gemini
# ---------------------------------------------------------------------------

def get_red_provider() -> str:
    raw = (
        os.environ.get("RED_TEAM_PROVIDER")
        or os.environ.get("LLM_PROVIDER")
        or "openai"
    ).strip().lower()
    if raw in {"gemini", "google", "adk"}:
        return PROVIDER_GEMINI
    return PROVIDER_OPENAI


def get_red_model() -> str:
    """Model Red Team từ .env (cùng cho default + advance)."""
    if get_red_provider() == PROVIDER_GEMINI:
        return (
            os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
            or DEFAULT_GEMINI_MODEL
        )
    return (
        os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
        or DEFAULT_OPENAI_MODEL
    )


def get_red_model_default() -> str:
    """Alias — Red dùng cùng model .env."""
    return get_red_model()


def get_red_model_advance() -> str:
    """Alias — Red Advance dùng cùng model .env."""
    return get_red_model()


def get_openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()


def red_openai_client_kwargs() -> dict:
    return {"api_key": get_openai_api_key() or None}


def red_provider_label(tier: str = "advance") -> str:
    # tier giữ để tương thích call site; cả hai agent cùng model .env
    _ = tier
    return f"{get_red_provider()}:{get_red_model()}"


def red_uses_openai_sdk() -> bool:
    return get_red_provider() == PROVIDER_OPENAI


def red_uses_gemini() -> bool:
    return get_red_provider() == PROVIDER_GEMINI


# ---------------------------------------------------------------------------
# Backward-compatible aliases (mean RED TEAM — used by attack JSON / grade)
# ---------------------------------------------------------------------------

def get_llm_provider() -> str:
    return get_red_provider()


def get_model_name() -> str:
    """Model khai trong attack_results — khớp .env lúc chạy CP4."""
    return get_red_model()


def uses_openai_sdk() -> bool:
    """Deprecated name: True when Red Team uses OpenAI SDK (not Gemini ADK)."""
    return red_uses_openai_sdk()


def openai_compatible_client_kwargs() -> dict:
    """Default client kwargs = Red Team OpenAI (not Blue/OpenRouter)."""
    return red_openai_client_kwargs()


def provider_label() -> str:
    return red_provider_label()


def is_harder_model() -> bool:
    """True nếu .env đang trỏ model khó (luna / 3.8) — tuỳ chọn, không phải tên agent."""
    m = get_red_model().lower()
    if m in {DEFAULT_OPENAI_MODEL.lower(), DEFAULT_GEMINI_MODEL.lower()}:
        return False
    hard = {
        HARD_OPENAI_MODEL.lower(),
        HARD_GEMINI_MODEL.lower(),
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-4o",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-2.5-pro",
    }
    if m in hard:
        return True
    return any(x in m for x in ("gpt-5.6", "pro", "gemini-3.8", "gemini-3.7"))


def setup_api_key():
    """Ensure keys for Blue (OpenRouter) + Red / Red Advance (OpenAI or Gemini)."""
    if get_blue_provider() == PROVIDER_GROQ:
        if not os.environ.get("GROQ_API_KEY", "").strip():
            os.environ["GROQ_API_KEY"] = input("Enter Groq API Key (Blue): ").strip()
    elif not get_openrouter_api_key():
        os.environ["OPENROUTER_API_KEY"] = input(
            "Enter OpenRouter API Key (Blue): "
        ).strip()
    print(f"Blue  — {blue_provider_label()}")

    red = get_red_provider()
    model = get_red_model()
    if red == PROVIDER_GEMINI:
        if not os.environ.get("GOOGLE_API_KEY", "").strip():
            os.environ["GOOGLE_API_KEY"] = input("Enter Google API Key (Red): ").strip()
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"
        print(f"Red / Red Advance  — gemini:{model}")
    else:
        if not get_openai_api_key():
            os.environ["OPENAI_API_KEY"] = input("Enter OpenAI API Key (Red): ").strip()
        print(f"Red / Red Advance  — openai:{model}")

    print(
        "Bonus: chọn một — Red tối đa +5 (B1) hoặc Red Advance tối đa +10 (B2)."
    )
    if is_harder_model():
        print(
            f"Model khó ({model}) — tuỳ chọn; không đổi tên agent. "
            f"(Gợi ý: {HARD_OPENAI_MODEL} / {HARD_GEMINI_MODEL})"
        )


ALLOWED_TOPICS = [
    "banking", "account", "transaction", "transfer",
    "loan", "interest", "savings", "credit",
    "deposit", "withdrawal", "balance", "payment",
    "tai khoan", "giao dich", "tiet kiem", "lai suat",
    "chuyen tien", "the tin dung", "so du", "vay",
    "ngan hang", "atm",
]

BLOCKED_TOPICS = [
    "hack", "exploit", "weapon", "drug", "illegal",
    "violence", "gambling", "bomb", "kill", "steal",
]
