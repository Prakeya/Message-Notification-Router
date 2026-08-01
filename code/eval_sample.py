import csv
from pathlib import Path

from ingest import MediaProcessor, load_csv_rows
from retrieval import Retriever
from features import FeatureEngine
from decision import DecisionEngine

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"


def main() -> None:
    samples = load_csv_rows(DATASET_DIR / "sample_messages.csv")
    message_history = load_csv_rows(DATASET_DIR / "message_history.csv")
    message_events = load_csv_rows(DATASET_DIR / "message_events.csv")
    users = load_csv_rows(DATASET_DIR / "users.csv")
    groups = load_csv_rows(DATASET_DIR / "groups.csv")
    group_members = load_csv_rows(DATASET_DIR / "group_members.csv")
    business_accounts = load_csv_rows(DATASET_DIR / "business_accounts.csv")
    user_business_history = load_csv_rows(DATASET_DIR / "user_business_history.csv")
    images = load_csv_rows(DATASET_DIR / "images.csv")
    voice_notes = load_csv_rows(DATASET_DIR / "voice_notes.csv")
    daily_notification_summary = load_csv_rows(DATASET_DIR / "daily_notification_summary.csv")

    image_lookup = {row.get("image_id"): row for row in images}
    voice_lookup = {row.get("voice_note_id"): row for row in voice_notes}

    processor = MediaProcessor(str(DATASET_DIR))
    retriever = Retriever(message_history, message_events)
    feature_engine = FeatureEngine(users, groups, group_members, business_accounts, user_business_history, message_events, daily_notification_summary)
    decision_engine = DecisionEngine(debug_path=None)

    action_correct = 0
    type_correct = 0
    both_correct = 0
    mismatches = []

    for message in samples:
        media_info = processor.analyze(message, image_lookup, voice_lookup)
        evidence = retriever.retrieve(message, top_k=2)
        features = feature_engine.build_features(message, evidence, media_info)
        decision = decision_engine.decide(message, features, evidence, media_info)

        pred_action = decision.get("action")
        pred_type = decision.get("message_type")
        gt_action = message.get("action")
        gt_type = message.get("message_type")

        a_ok = pred_action == gt_action
        t_ok = pred_type == gt_type
        if a_ok:
            action_correct += 1
        if t_ok:
            type_correct += 1
        if a_ok and t_ok:
            both_correct += 1
        else:
            mismatches.append({
                "message_id": message.get("message_id"),
                "media_type": message.get("media_type"),
                "extraction_source": media_info.get("extraction_source"),
                "pred_action": pred_action, "gt_action": gt_action,
                "pred_type": pred_type, "gt_type": gt_type,
                "unified_text": media_info.get("unified_text", "")[:120],
            })

    n = len(samples)
    print(f"action correct: {action_correct}/{n}")
    print(f"message_type correct: {type_correct}/{n}")
    print(f"both correct: {both_correct}/{n}")
    print()
    print("Mismatches:")
    for m in mismatches:
        print(m)

    print()
    print("Voice-note rows detail:")
    for message in samples:
        if message.get("media_type") == "voice":
            media_info = processor.analyze(message, image_lookup, voice_lookup)
            print({
                "message_id": message.get("message_id"),
                "media_id": message.get("media_id"),
                "extraction_source": media_info.get("extraction_source"),
                "unified_text": media_info.get("unified_text", "")[:150],
                "gt_action": message.get("action"),
                "gt_type": message.get("message_type"),
            })


if __name__ == "__main__":
    main()
