import os
import time
import requests
from datetime import datetime

def send_telegram_message(bot_token, chat_id, text):
    """Send a single message to a Telegram chat using the Bot API."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False

def send_telegram_messages(bot_token, chat_id, messages):
    """Send multiple messages sequentially with a small delay."""
    all_ok = True
    for msg in messages:
        ok = send_telegram_message(bot_token, chat_id, msg)
        if not ok:
            all_ok = False
        time.sleep(0.5)  # small delay to preserve message order
    return all_ok

def format_vp_messages(monday, friday, symbol, va_results):
    """
    Formats the volume profile results into separate Telegram messages.

    Returns a list of messages:
      1. Header:  📅 Week: Feb 23 - Feb 27, 2026
      2. Per VA:  ```
                  70% Value Area
                  VAH: 6919.25
                  POC: 6882.00
                  VAL: 6852.25
                  ```
    """
    week_str = f"{monday.strftime('%b %d')} - {friday.strftime('%b %d, %Y')}"
    header = f"📅 Week: {week_str}"

    messages = [header]
    for va, (poc, vah, val) in va_results.items():
        block = (
            f"```\n"
            f"{va:.0f}% Value Area\n"
            f"VAH: {vah:.2f}\n"
            f"POC: {poc:.2f}\n"
            f"VAL: {val:.2f}\n"
            f"```"
        )
        messages.append(block)

    return messages
