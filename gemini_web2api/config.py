"""Configuration management."""
import json
import os

DEFAULT_CONFIG = {
    "port": 8081,
    "host": "0.0.0.0",
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_timeout_sec": 60,
    "gemini_bl": "boq_assistant-bard-web-server_20260716.08_p0",
    "auth_user": None,
    "xsrf_token": None,
    "default_model": "gemini-3.8-flash",
    "log_requests": True,
    "cookie_file": None,
    "proxy": None,
    "api_keys": [],
    "temporary_chats": False,
}

CONFIG = dict(DEFAULT_CONFIG)


def load_config(path: str = None):
    """Load config from JSON file and environment variables."""
    if path and os.path.exists(path):
        with open(path) as f:
            CONFIG.update(json.load(f))
    env_keys = os.environ.get("API_KEYS") or os.environ.get("API_KEY")
    if env_keys:
        CONFIG["api_keys"] = [k.strip() for k in env_keys.split(",") if k.strip()]
    env_model = os.environ.get("DEFAULT_MODEL") or os.environ.get("GEMINI_MODEL")
    if env_model:
        CONFIG["default_model"] = env_model.strip()
    env_timeout = os.environ.get("REQUEST_TIMEOUT_SEC")
    if env_timeout and env_timeout.isdigit():
        CONFIG["request_timeout_sec"] = int(env_timeout)
    return CONFIG


def find_config():
    """Search for config file in standard locations."""
    for p in ["./config.json", os.path.expanduser("~/.config/gemini-web2api/config.json")]:
        if os.path.exists(p):
            return p
    return None
