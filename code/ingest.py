import csv
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


class MediaProcessor:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.ocr_available = shutil.which("tesseract") is not None and pytesseract is not None and Image is not None
        self.asr_available = bool(os.getenv("OPENAI_API_KEY"))

    def analyze(self, message: Dict[str, str], image_lookup: Dict[str, Dict[str, str]], voice_lookup: Dict[str, Dict[str, str]]) -> Dict[str, object]:
        text = (message.get("message_text") or "").strip()
        if not message.get("media_type"):
            unified_text = self._normalize(text or "")
            return {
                "unified_text": unified_text,
                "media_summary": unified_text or "text message",
                "ocr_text": "",
                "visual_summary": "",
                "contains_payment_request": False,
                "contains_link_or_qr": False,
                "contains_urgency_language": False,
                "summary": "",
                "entities": [],
                "contains_payment_or_money_mention": False,
                "media_available": False,
                "extraction_source": "none",
            }

        if message.get("media_type") == "image":
            media_id = message.get("media_id") or ""
            media_row = image_lookup.get(media_id, {})
            file_path = media_row.get("file_path", "")
            path = self.dataset_dir / file_path if file_path else None
            if path and path.exists() and self.ocr_available:
                try:
                    with Image.open(path) as img:
                        extracted = pytesseract.image_to_string(img)
                    summary = self._normalize(extracted or text or f"image {media_id}")
                    contains_payment_request = bool(re.search(r"payment|otp|verify|pay|card|bank|booking|delivery|refund|wallet", summary.lower()))
                    contains_link_or_qr = bool(re.search(r"link|qr|bit\.ly|http|https|scan", summary.lower()))
                    contains_urgency_language = bool(re.search(r"urgent|today|tomorrow|now|deadline|before|by [0-9]|immediately|eod|tonight", summary.lower()))
                    return {
                        "unified_text": self._normalize(summary),
                        "media_summary": self._normalize(summary),
                        "ocr_text": self._normalize(summary),
                        "visual_summary": "media image with text context",
                        "contains_payment_request": contains_payment_request,
                        "contains_link_or_qr": contains_link_or_qr,
                        "contains_urgency_language": contains_urgency_language,
                        "summary": self._normalize(summary),
                        "entities": [],
                        "contains_payment_or_money_mention": contains_payment_request,
                        "media_available": True,
                        "extraction_source": "ocr",
                    }
                except Exception:
                    # OCR extraction failed or is unavailable; continue with the metadata-based fallback summary.
                    summary = text or f"image {media_id}"
            else:
                summary = text or f"image {media_id}"
            return {
                "unified_text": self._normalize(summary),
                "media_summary": self._normalize(summary),
                "ocr_text": "",
                "visual_summary": "media fallback",
                "contains_payment_request": bool(re.search(r"payment|otp|verify|pay|card|bank|booking|delivery|refund|wallet", summary.lower())),
                "contains_link_or_qr": bool(re.search(r"link|qr|bit\.ly|http|https|scan", summary.lower())),
                "contains_urgency_language": bool(re.search(r"urgent|today|tomorrow|now|deadline|before|by [0-9]|immediately|eod|tonight", summary.lower())),
                "summary": self._normalize(summary),
                "entities": [],
                "contains_payment_or_money_mention": bool(re.search(r"payment|otp|verify|pay|card|bank|wallet|refund|fee|money|cash", summary.lower())),
                "media_available": False,
                "extraction_source": "fallback",
            }

        if message.get("media_type") == "voice":
            media_id = message.get("media_id") or ""
            media_row = voice_lookup.get(media_id, {})
            file_path = media_row.get("file_path", "")
            path = self.dataset_dir / file_path if file_path else None
            if path and path.exists() and self.asr_available and requests is not None:
                try:
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        raise RuntimeError("OPENAI_API_KEY not configured")
                    endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/audio/transcriptions")
                    model = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
                    with path.open("rb") as handle:
                        files = {"file": (path.name, handle, "application/octet-stream")}
                        data = {"model": model}
                        response = requests.post(endpoint, headers={"Authorization": f"Bearer {api_key}"}, files=files, data=data, timeout=60)
                        response.raise_for_status()
                    payload = response.json()
                    summary = self._normalize((payload.get("text") or "") or text or f"voice note {media_id}")
                    contains_payment_request = bool(re.search(r"payment|otp|verify|pay|card|bank|booking|delivery|refund|wallet", summary.lower()))
                    contains_link_or_qr = bool(re.search(r"link|qr|bit\.ly|http|https|scan", summary.lower()))
                    contains_urgency_language = bool(re.search(r"urgent|today|tomorrow|now|deadline|before|by [0-9]|immediately|eod|tonight", summary.lower()))
                    return {
                        "unified_text": self._normalize(summary),
                        "media_summary": self._normalize(summary),
                        "ocr_text": "",
                        "visual_summary": "voice note",
                        "contains_payment_request": contains_payment_request,
                        "contains_link_or_qr": contains_link_or_qr,
                        "contains_urgency_language": contains_urgency_language,
                        "summary": self._normalize(summary),
                        "entities": [],
                        "contains_payment_or_money_mention": bool(re.search(r"payment|otp|verify|pay|card|bank|wallet|refund|fee|money|cash", summary.lower())),
                        "media_available": True,
                        "extraction_source": "asr",
                    }
                except Exception:
                    summary = text or f"voice message {media_id}"
            else:
                summary = text or f"voice message {media_id}"
            return {
                "unified_text": self._normalize(summary),
                "media_summary": self._normalize(summary),
                "ocr_text": "",
                "visual_summary": "audio fallback",
                "contains_payment_request": False,
                "contains_link_or_qr": False,
                "contains_urgency_language": bool(re.search(r"urgent|today|tomorrow|now|deadline|before|immediately|tonight", summary.lower())),
                "summary": self._normalize(summary),
                "entities": [],
                "contains_payment_or_money_mention": bool(re.search(r"payment|otp|verify|pay|card|bank|wallet|refund|fee|money|cash", summary.lower())),
                "media_available": False,
                "extraction_source": "fallback",
            }

        return {
            "unified_text": self._normalize(text or ""),
            "media_summary": self._normalize(text or ""),
            "ocr_text": "",
            "visual_summary": "",
            "contains_payment_request": False,
            "contains_link_or_qr": False,
            "contains_urgency_language": False,
            "summary": "",
            "entities": [],
            "contains_payment_or_money_mention": False,
            "media_available": False,
            "extraction_source": "none",
        }

    def _normalize(self, text: str) -> str:
        text = (text or "").replace("\n", " ").strip()
        text = re.sub(r"\s+", " ", text)
        return text


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
