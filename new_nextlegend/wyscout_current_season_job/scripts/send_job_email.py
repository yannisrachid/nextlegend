from __future__ import annotations

import argparse
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send job status email via SMTP.")
    parser.add_argument("--subject", required=True, help="Email subject.")
    parser.add_argument("--body-file", help="Path to plain-text body file.")
    parser.add_argument("--log-file", help="Path to a log file to attach.")
    parser.add_argument(
        "--max-log-lines",
        type=int,
        default=200,
        help="Max number of log lines included in the body tail (default: 200).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if SMTP settings are missing.",
    )
    return parser.parse_args(argv)


def _parse_recipients(raw: str) -> list[str]:
    cleaned = raw.replace(";", ",")
    recipients = [part.strip() for part in cleaned.split(",")]
    return [value for value in recipients if value]


def _load_body(path: str | None) -> str:
    if not path:
        return ""
    body_path = Path(path).expanduser()
    if not body_path.exists():
        return ""
    return body_path.read_text(encoding="utf-8")


def _tail_lines(path: str | None, max_lines: int) -> str:
    if not path:
        return ""
    log_path = Path(path).expanduser()
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_lines <= 0:
        return ""
    return "\n".join(lines[-max_lines:])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587") or "587")
    smtp_user = os.getenv("SMTP_USERNAME", "").strip()
    smtp_pass = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_from = os.getenv("SMTP_FROM", "").strip() or smtp_user
    smtp_to_raw = os.getenv("SMTP_TO", "").strip()
    smtp_use_tls = os.getenv("SMTP_USE_TLS", "1").strip().lower() not in {"0", "false", "no"}

    recipients = _parse_recipients(smtp_to_raw)
    missing = [
        key
        for key, value in (
            ("SMTP_HOST", smtp_host),
            ("SMTP_FROM", smtp_from),
            ("SMTP_TO", ",".join(recipients)),
        )
        if not value
    ]
    if missing:
        print(f"[EMAIL] skip: missing SMTP settings: {', '.join(missing)}")
        return 1 if args.strict else 0

    body = _load_body(args.body_file)
    log_tail = _tail_lines(args.log_file, args.max_log_lines)
    if log_tail:
        body = (body.rstrip() + "\n\n---\nLog tail:\n\n" + log_tail).strip() + "\n"
    elif body:
        body = body.rstrip() + "\n"
    else:
        body = "Job notification.\n"

    msg = EmailMessage()
    msg["Subject"] = args.subject
    msg["From"] = smtp_from
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    if smtp_use_tls:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls(context=context)
            if smtp_user:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            if smtp_user:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)

    print(f"[EMAIL] sent to {len(recipients)} recipient(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
