import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ingest import MediaProcessor, load_csv_rows
from retrieval import Retriever
from features import FeatureEngine
from decision import DecisionEngine
from safety import SafetyValidator


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"


def main() -> None:
    messages = load_csv_rows(DATASET_DIR / "messages.csv")
    message_history = load_csv_rows(DATASET_DIR / "message_history.csv")
    message_events = load_csv_rows(DATASET_DIR / "message_events.csv")
    users = load_csv_rows(DATASET_DIR / "users.csv")
    groups = load_csv_rows(DATASET_DIR / "groups.csv")
    group_members = load_csv_rows(DATASET_DIR / "group_members.csv")
    business_accounts = load_csv_rows(DATASET_DIR / "business_accounts.csv")
    user_business_history = load_csv_rows(DATASET_DIR / "user_business_history.csv")
    images = load_csv_rows(DATASET_DIR / "images.csv")
    voice_notes = load_csv_rows(DATASET_DIR / "voice_notes.csv")

    image_lookup = {row.get("image_id"): row for row in images}
    voice_lookup = {row.get("voice_note_id"): row for row in voice_notes}

    processor = MediaProcessor(str(DATASET_DIR))
    retriever = Retriever(message_history, message_events)
    feature_engine = FeatureEngine(users, groups, group_members, business_accounts, user_business_history, message_events)
    debug_path = ROOT / "logs" / "debug_predictions.jsonl"
    os.makedirs(ROOT / "logs", exist_ok=True)
    if debug_path.exists():
        debug_path.unlink()
    decision_engine = DecisionEngine(debug_path=str(debug_path))
    validator = SafetyValidator(message_history)

    predictions: List[Dict[str, object]] = []
    for message in messages:
        media_info = processor.analyze(message, image_lookup, voice_lookup)
        evidence = retriever.retrieve(message, {}, top_k=2)
        features = feature_engine.build_features(message, {}, evidence, media_info)
        decision = decision_engine.decide(message, features, evidence, media_info)
        prediction = {
            "message_id": message.get("message_id"),
            "action": decision.get("action"),
            "message_type": decision.get("message_type"),
            "reason": decision.get("reason"),
            "confidence": decision.get("confidence"),
            "evidence_message_ids": decision.get("evidence_message_ids"),
        }
        predictions.append(prediction)

    validator.validate(predictions, messages)

    out_path = DATASET_DIR / "output.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Wrote {len(predictions)} predictions to {out_path}")


if __name__ == "__main__":
    main()
