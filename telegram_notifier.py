import os
import requests
from datetime import datetime

def send_telegram_message(bot_token, chat_id, text):
    """Send a message to a Telegram chat using the Bot API."""
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

def format_vp_message(monday, friday, symbol, va_results):
    """
    Formats the volume profile results into the requested Telegram format.
    
    Format:
    📅 Week: Feb 16 - Feb 20, 2026
    
    ```
    70% Value Area
    VAH:  6913.00
    POC:  6874.75
    VAL:  6851.75
    ```
    
    ```
    68% Value Area
    VAH:  6911.00
    POC:  6874.75
    VAL:  6851.75
    ```
    """
    week_str = f"{monday.strftime('%b %d')} - {friday.strftime('%b %d, %Y')}"
    header = f"📅 *Week: {week_str}*\n\n"
    
    blocks = []
    for va, (poc, vah, val) in va_results.items():
        block = (
            f"```\n"
            f"{va:.0f}% Value Area\n"
            f"VAH:  {vah:.2f}\n"
            f"POC:  {poc:.2f}\n"
            f"VAL:  {val:.2f}\n"
            f"```"
        )
        blocks.append(block)
    
    return header + "\n".join(blocks)
