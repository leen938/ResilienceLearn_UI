from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None  # type: ignore[assignment]
    _openai_import_error = e
else:
    _openai_import_error = None


def _truthy(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    dotenv_path = root / ".env"
    parsed = dotenv_values(dotenv_path)
    loaded = load_dotenv(dotenv_path=dotenv_path, override=False)

    use_openai_raw = os.environ.get("USE_OPENAI_CHAT", "")
    llm_enabled_raw = os.environ.get("LLM_SUPPORT_ENABLED", "")
    llm_provider_raw = os.environ.get("LLM_PROVIDER", "")
    use_openai = _truthy(use_openai_raw) or (_truthy(llm_enabled_raw) and llm_provider_raw.strip().lower() == "openai")
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    api_key_present = bool(api_key)
    model = (os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL_NAME") or "gpt-4o-mini").strip()

    print(f"repo_root={root}")
    print(f"dotenv_path={dotenv_path} exists={dotenv_path.is_file()} loaded={loaded}")
    print(f"dotenv_parsed_keys={sorted([str(k) for k in parsed.keys()])}")
    print(
        f"USE_OPENAI_CHAT raw={use_openai_raw!r} "
        f"LLM_SUPPORT_ENABLED raw={llm_enabled_raw!r} "
        f"LLM_PROVIDER raw={llm_provider_raw!r} "
        f"parsed_use_openai={use_openai}"
    )
    print(f"OPENAI_API_KEY present={api_key_present} length={len(api_key)}")
    print(f"OPENAI_MODEL={model!r}")
    print(f"openai_sdk_available={OpenAI is not None}")

    if OpenAI is None:
        print(f"OpenAI import error: {_openai_import_error!r}")
        return 2

    if not use_openai:
        print("Refusing to call OpenAI because USE_OPENAI_CHAT is not truthy.")
        return 3

    if not api_key_present:
        print("Refusing to call OpenAI because OPENAI_API_KEY is missing/empty.")
        return 4

    client = OpenAI(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a warm, emotionally intelligent supportive chat companion for students. "
                "Be natural, human, and conversational. Keep replies short-to-medium. "
                "You are NOT a therapist or doctor. Do not diagnose or make medical claims."
            ),
        },
        {"role": "user", "content": "Hi. I’m stressed about exams and I can’t sleep. Can you talk with me?"},
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.8,
            max_tokens=120,
        )
        text = (resp.choices[0].message.content or "").strip()
        print("OpenAI call: SUCCESS")
        print("Reply (first 300 chars):")
        print(text[:300])
        return 0
    except Exception as e:
        print("OpenAI call: FAILED")
        print(f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

