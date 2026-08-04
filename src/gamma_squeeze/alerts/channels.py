"""Notification channels: WebSocket (app-level), Email, SMS, Discord, Slack, Webhook."""

from __future__ import annotations

import base64
import json
import os
import smtplib
import ssl
import urllib.error
import urllib.parse
import urllib.request
from email.message import EmailMessage
from typing import Any

from gamma_squeeze.config import load_env

CHANNELS: tuple[str, ...] = (
    "WebSocket",
    "Email",
    "SMS",
    "Discord",
    "Slack",
    "Webhook",
)


def _post_json(url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "GammaSqueezeAlertEngine/1.0", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {"ok": True, "status": resp.status, "body": body[:500]}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": exc.read().decode("utf-8", errors="replace")[:500]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def channel_config() -> dict[str, Any]:
    load_env()
    return {
        "email": {
            "enabled": bool(os.getenv("ALERT_EMAIL_TO") and os.getenv("ALERT_SMTP_HOST")),
            "to": os.getenv("ALERT_EMAIL_TO", ""),
            "from": os.getenv("ALERT_EMAIL_FROM", "alerts@gamma-squeeze.local"),
            "smtp_host": os.getenv("ALERT_SMTP_HOST", ""),
            "smtp_port": int(os.getenv("ALERT_SMTP_PORT", "587") or 587),
        },
        "sms": {
            "enabled": bool(
                os.getenv("ALERT_TWILIO_ACCOUNT_SID")
                and os.getenv("ALERT_TWILIO_AUTH_TOKEN")
                and os.getenv("ALERT_SMS_TO")
            ),
            "to": os.getenv("ALERT_SMS_TO", ""),
            "from": os.getenv("ALERT_SMS_FROM", ""),
        },
        "discord": {
            "enabled": bool(os.getenv("ALERT_DISCORD_WEBHOOK_URL")),
            "url_set": bool(os.getenv("ALERT_DISCORD_WEBHOOK_URL")),
        },
        "slack": {
            "enabled": bool(os.getenv("ALERT_SLACK_WEBHOOK_URL")),
            "url_set": bool(os.getenv("ALERT_SLACK_WEBHOOK_URL")),
        },
        "webhook": {
            "enabled": bool(os.getenv("ALERT_WEBHOOK_URL")),
            "url_set": bool(os.getenv("ALERT_WEBHOOK_URL")),
        },
        "websocket": {"enabled": True, "path": "/v1/ws/alerts"},
    }


def _format_text(symbol: str, alerts: list[dict[str, Any]]) -> str:
    lines = [f"Gamma Squeeze Alerts — {symbol}", ""]
    for a in alerts:
        lines.append(f"[{str(a.get('severity', 'info')).upper()}] {a.get('type')}: {a.get('message')}")
    return "\n".join(lines)


def send_email(symbol: str, alerts: list[dict[str, Any]]) -> dict[str, Any]:
    load_env()
    cfg = channel_config()["email"]
    if not cfg["enabled"]:
        return {"channel": "Email", "skipped": True, "reason": "not_configured"}
    msg = EmailMessage()
    msg["Subject"] = f"[Gamma Alerts] {symbol} — {len(alerts)} event(s)"
    msg["From"] = cfg["from"]
    msg["To"] = cfg["to"]
    msg.set_content(_format_text(symbol, alerts))
    user = os.getenv("ALERT_SMTP_USER", "")
    password = os.getenv("ALERT_SMTP_PASSWORD", "")
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"], timeout=20) as smtp:
            smtp.starttls(context=context)
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)
        return {"channel": "Email", "ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"channel": "Email", "ok": False, "error": str(exc)}


def send_sms(symbol: str, alerts: list[dict[str, Any]]) -> dict[str, Any]:
    load_env()
    cfg = channel_config()["sms"]
    if not cfg["enabled"]:
        return {"channel": "SMS", "skipped": True, "reason": "not_configured"}
    sid = os.getenv("ALERT_TWILIO_ACCOUNT_SID", "")
    token = os.getenv("ALERT_TWILIO_AUTH_TOKEN", "")
    from_num = os.getenv("ALERT_SMS_FROM", "")
    to_num = cfg["to"]
    body = _format_text(symbol, alerts)[:1500]
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    form = urllib.parse.urlencode({"From": from_num, "To": to_num, "Body": body}).encode()
    req = urllib.request.Request(url, data=form, method="POST")
    creds = f"{sid}:{token}".encode()
    req.add_header("Authorization", "Basic " + base64.b64encode(creds).decode())
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return {"channel": "SMS", "ok": True, "status": resp.status}
    except Exception as exc:  # noqa: BLE001
        return {"channel": "SMS", "ok": False, "error": str(exc)}


def send_discord(symbol: str, alerts: list[dict[str, Any]]) -> dict[str, Any]:
    load_env()
    url = os.getenv("ALERT_DISCORD_WEBHOOK_URL", "").strip()
    if not url:
        return {"channel": "Discord", "skipped": True, "reason": "not_configured"}
    content = _format_text(symbol, alerts)
    if len(content) > 1900:
        content = content[:1900] + "…"
    result = _post_json(url, {"content": content})
    return {"channel": "Discord", **result}


def send_slack(symbol: str, alerts: list[dict[str, Any]]) -> dict[str, Any]:
    load_env()
    url = os.getenv("ALERT_SLACK_WEBHOOK_URL", "").strip()
    if not url:
        return {"channel": "Slack", "skipped": True, "reason": "not_configured"}
    text = _format_text(symbol, alerts)
    result = _post_json(url, {"text": text})
    return {"channel": "Slack", **result}


def send_webhook(symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
    del symbol
    load_env()
    url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if not url:
        return {"channel": "Webhook", "skipped": True, "reason": "not_configured"}
    result = _post_json(url, payload)
    return {"channel": "Webhook", **result}


def dispatch_alerts(
    symbol: str,
    alert_payload: dict[str, Any],
    *,
    channels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fan-out fired alerts to configured channels (WebSocket is app-layer)."""
    alerts = alert_payload.get("alerts") or []
    if not alerts or not alert_payload.get("fired"):
        return [{"channel": "all", "skipped": True, "reason": "no_alerts"}]

    wanted = {c.lower() for c in (channels or list(CHANNELS))}
    results: list[dict[str, Any]] = []
    if "websocket" in wanted:
        results.append({"channel": "WebSocket", "ok": True, "note": "streamed by /v1/ws/alerts subscribers"})
    if "email" in wanted:
        results.append(send_email(symbol, alerts))
    if "sms" in wanted:
        results.append(send_sms(symbol, alerts))
    if "discord" in wanted:
        results.append(send_discord(symbol, alerts))
    if "slack" in wanted:
        results.append(send_slack(symbol, alerts))
    if "webhook" in wanted:
        results.append(send_webhook(symbol, alert_payload))
    return results
