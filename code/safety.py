from typing import Dict, List


class SafetyValidator:
    def __init__(self, message_history_rows: List[Dict[str, str]]):
        self.valid_message_ids = {row.get("message_id") for row in message_history_rows}
        self.allowed_actions = {"notify", "digest", "mute"}
        self.allowed_types = {"personal", "urgent", "event", "payment", "business_update", "promotion", "greeting", "forward", "spam", "scam", "unknown"}

    def validate(self, predictions: List[Dict[str, object]], messages: List[Dict[str, str]]) -> None:
        if len(predictions) != len(messages):
            raise ValueError(f"Expected {len(messages)} predictions, got {len(predictions)}")
        for message, prediction in zip(messages, predictions):
            action = prediction.get("action")
            message_type = prediction.get("message_type")
            evidence = prediction.get("evidence_message_ids", "")
            confidence = prediction.get("confidence")
            if action not in self.allowed_actions:
                raise ValueError(f"Invalid action for {message.get('message_id')}: {action}")
            if message_type not in self.allowed_types:
                raise ValueError(f"Invalid type for {message.get('message_id')}: {message_type}")
            if evidence != "none":
                ids = [item.strip() for item in evidence.split(';') if item.strip()]
                unknown = [item for item in ids if item not in self.valid_message_ids]
                if unknown:
                    raise ValueError(f"Invalid evidence IDs for {message.get('message_id')}: {unknown}")
            try:
                conf = float(confidence)
            except Exception as exc:
                raise ValueError(f"Invalid confidence for {message.get('message_id')}: {confidence}") from exc
            if not (0.0 <= conf <= 1.0):
                raise ValueError(f"Confidence out of range for {message.get('message_id')}: {conf}")
