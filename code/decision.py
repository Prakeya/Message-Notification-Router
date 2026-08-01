import json
import os
import re
from typing import Dict, List, Optional

import requests


ACTION_ORDER = {"notify": 0, "digest": 1, "mute": 2}


class DecisionEngine:
    def __init__(self, llm_client=None, debug_path: Optional[str] = None):
        self.llm_client = llm_client
        self.debug_path = debug_path

    def decide(self, message: Dict[str, str], features: Dict[str, object], evidence: List[Dict[str, object]], media_info: Dict[str, object]) -> Dict[str, object]:
        text = (message.get("message_text") or "").strip()
        if not text and media_info.get("media_summary"):
            text = media_info.get("media_summary")
        text = re.sub(r"\s+", " ", text)
        reason = ""
        action = "digest"
        message_type = "unknown"
        confidence = 0.7 * features.get("rule_certainty", 0.5) + 0.3 * features.get("evidence_strength", 0.0)

        # deterministic rules first
        rule_action, rule_message_type, rule_reason = self._deterministic_decision(message, text, features)
        action, message_type, reason = rule_action, rule_message_type, rule_reason

        llm_used = False
        llm_succeeded = False
        fallback_used = False
        top_signals = self._top_signals(features, message, text, evidence)
        conflict_count = self._conflict_count(message, text, features, evidence, media_info)
        if conflict_count >= 2:
            llm_used = True
            last_error = None
            for attempt in range(2):
                try:
                    llm_payload = self._call_llm(message, text, features, evidence, media_info)
                    validated = self._validated_llm_payload(llm_payload, evidence)
                    if validated:
                        action = validated["action"]
                        message_type = validated["message_type"]
                        reason = validated["reason"]
                        confidence = self._final_confidence(action, message_type, features, evidence, confidence, True)
                        llm_succeeded = True
                    else:
                        raise RuntimeError("invalid llm response")
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 0:
                        continue
                    fallback_used = True
                    action, message_type, reason = self._fallback_decision(message, text, features, evidence)
                    confidence = self._final_confidence(action, message_type, features, evidence, confidence, False)
            if llm_succeeded:
                fallback_used = False
        else:
            confidence = self._final_confidence(action, message_type, features, evidence, confidence, False)

        evidence_ids = self._evidence_ids(evidence)
        decision = {
            "action": action,
            "message_type": message_type,
            "reason": reason,
            "confidence": round(confidence, 3),
            "evidence_message_ids": evidence_ids,
        }
        self._append_debug(message, features, evidence, media_info, decision, top_signals, llm_used, llm_succeeded, fallback_used, conflict_count)
        return decision

    def _deterministic_decision(self, message: Dict[str, str], text: str, features: Dict[str, object]) -> tuple:
        scam_score = float(features.get("scam_score", 0.0) or 0.0)
        if features.get("scam_band") == "High" and scam_score >= 0.88:
            return "mute", "scam", "high scam risk indicators"
        if features.get("urgency_band") == "High" and (features.get("role") == "admin" or features.get("business_verified")) and self._has_explicit_window(text):
            return "notify", "urgent", "trusted group admin"
        if self._is_business_transaction(message, text, features):
            return "notify", "payment", "transactional business request"
        if self._is_marketing(message, text, features):
            if features.get("allows_promotions"):
                return "digest", "promotion", "user opted into promotions"
            return "mute", "promotion", "user opted out of promotions"
        if features.get("is_duplicate"):
            return "digest", "forward", "sender has a pattern of repeated forwards"
        if features.get("group_muted") and not features.get("is_direct_mention") and features.get("urgency_band") != "High":
            return "mute", "unknown", "muted group without direct mention"
        if self._is_forward_pattern(text):
            return "mute", "greeting", "forward or greeting pattern"
        if self._looks_personal(message, text):
            return "notify", "personal", "personal context with a direct request"
        if self._looks_business(text):
            return "digest", "business_update", "routine business update"
        if self._looks_promo(text):
            return "digest", "promotion", "marketing content"
        if self._looks_event(text):
            if features.get("urgency_band") == "High":
                return "notify", "event", "scheduled activity with a clear deadline"
            return "digest", "event", "scheduled activity notice"
        if self._looks_spam(text):
            return "mute", "spam", "bulk-style unsolicited content"
        if self._looks_payment(text):
            return "notify", "payment", "new sender with a payment request"
        if self._looks_urgent(text):
            return "notify", "urgent", "time-sensitive content"
        return "digest", "unknown", "no strong routing signal"

    def _fallback_decision(self, message: Dict[str, str], text: str, features: Dict[str, object], evidence: List[Dict[str, object]]) -> tuple:
        rule_action, rule_message_type, rule_reason = self._deterministic_decision(message, text, features)
        if rule_action in {"notify", "digest", "mute"} and rule_message_type in {"personal", "urgent", "event", "payment", "business_update", "promotion", "greeting", "forward", "spam", "scam", "unknown"}:
            return rule_action, rule_message_type, rule_reason
        return "digest", "unknown", rule_reason

    def _conflict_count(self, message: Dict[str, str], text: str, features: Dict[str, object], evidence: List[Dict[str, object]], media_info: Dict[str, object]) -> int:
        count = 0
        if features.get("trust_band") == "Medium":
            count += 1
        if features.get("scam_band") == "Medium":
            count += 1
        if self._ambiguous_evidence(evidence):
            count += 1
        if features.get("group_muted") and features.get("is_direct_mention"):
            count += 1
        if self._new_sender_plus_payment(message, text):
            count += 1
        if self._mixed_engagement_promotion(message, text, evidence):
            count += 1
        if self._media_mismatch(media_info, text):
            count += 1
        return count

    def _ambiguous_evidence(self, evidence: List[Dict[str, object]]) -> bool:
        if len(evidence) < 2:
            return False
        reactions = [str(item.get("reaction") or "") for item in evidence if item.get("reaction")]
        if len(set(reactions)) > 1:
            return True
        return len(evidence) >= 2 and evidence[0].get("similarity", 0.0) < 0.8

    def _new_sender_plus_payment(self, message: Dict[str, str], text: str) -> bool:
        return bool(message.get("sender_user_id") and self._looks_payment(text) and message.get("conversation_type") == "personal")

    def _mixed_engagement_promotion(self, message: Dict[str, str], text: str, evidence: List[Dict[str, object]]) -> bool:
        if not self._looks_promo(text):
            return False
        if not evidence:
            return True
        reactions = [str(item.get("reaction") or "") for item in evidence if item.get("reaction")]
        return len(set(reactions)) > 1

    def _media_mismatch(self, media_info: Dict[str, object], text: str) -> bool:
        if not media_info.get("media_summary"):
            return False
        return bool(media_info.get("media_summary") and not self._looks_urgent(text) and not self._looks_payment(text) and not self._looks_promo(text))

    def _call_llm(self, message: Dict[str, str], text: str, features: Dict[str, object], evidence: List[Dict[str, object]], media_info: Dict[str, object]) -> Optional[Dict[str, object]]:
        if os.getenv("LLM_FORCE_FAIL") == "1":
            raise RuntimeError("LLM_FORCE_FAIL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")
        endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": "You are the decision engine for a personalized WhatsApp notification router. Return JSON only."},
                {"role": "user", "content": self._llm_prompt(message, text, features, evidence, media_info)},
            ],
            "temperature": 0.1,
        }
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(content, str):
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                cleaned = cleaned.strip()
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    return None
        except Exception:
            raise
        return None

    def _llm_prompt(self, message: Dict[str, str], text: str, features: Dict[str, object], evidence: List[Dict[str, object]], media_info: Dict[str, object]) -> str:
        evidence_block = "\n".join([f"{item.get('message_id')}: {item.get('snippet')} | reaction={item.get('reaction')}" for item in evidence[:2]]) or "none"
        return (
            "You are routing an ambiguous WhatsApp message. Return ONLY raw JSON (no markdown fences, no commentary) with keys action, message_type, reason, confidence, evidence_message_ids.\n"
            "action must be exactly one of: notify, digest, mute\n"
            "message_type must be exactly one of: personal, urgent, event, payment, business_update, promotion, greeting, forward, spam, scam, unknown\n"
            f"Message: {text}\n"
            f"Context: trust_band={features.get('trust_band')} scam_band={features.get('scam_band')} group_muted={features.get('group_muted')} direct_mention={features.get('is_direct_mention')} urgency_band={features.get('urgency_band')}\n"
            f"Evidence: {evidence_block}\n"
            "Rules: if there is credible scam risk, action must be mute and message_type scam or spam."
        )

    def _validated_llm_payload(self, payload: Optional[Dict[str, object]], evidence: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if not payload:
            return None
        action = payload.get("action")
        message_type = payload.get("message_type")
        confidence = payload.get("confidence")
        evidence_ids = payload.get("evidence_message_ids")
        if action not in {"notify", "digest", "mute"}:
            return None
        if message_type not in {"personal", "urgent", "event", "payment", "business_update", "promotion", "greeting", "forward", "spam", "scam", "unknown"}:
            return None
        try:
            conf = float(confidence)
        except Exception:
            return None
        if not (0.0 <= conf <= 1.0):
            return None
        if evidence_ids and evidence_ids != "none":
            ids = [item.strip() for item in str(evidence_ids).split(';') if item.strip()]
            valid_ids = {item.get("message_id") for item in evidence if item.get("message_id")}
            if any(item not in valid_ids for item in ids):
                return None
        return payload

    def _top_signals(self, features: Dict[str, object], message: Dict[str, str], text: str, evidence: List[Dict[str, object]]) -> List[str]:
        signals = []
        if features.get("trust_band") == "Medium":
            signals.append("trust medium")
        if features.get("scam_band") == "Medium":
            signals.append("scam medium")
        if features.get("group_muted"):
            signals.append("muted group")
        if self._looks_payment(text):
            signals.append("payment cue")
        if evidence:
            signals.append("historical evidence")
        return signals[:4]

    def _append_debug(self, message: Dict[str, str], features: Dict[str, object], evidence: List[Dict[str, object]], media_info: Dict[str, object], decision: Dict[str, object], top_signals: List[str], llm_used: bool, llm_succeeded: bool, fallback_used: bool, conflict_count: int) -> None:
        if not self.debug_path:
            return
        os.makedirs(os.path.dirname(self.debug_path), exist_ok=True)
        payload = {
            "message_id": message.get("message_id"),
            "rule_scores": {
                "trust_score": features.get("trust_score"),
                "urgency_score": features.get("urgency_score"),
                "scam_score": features.get("scam_score"),
                "rule_certainty": features.get("rule_certainty"),
                "evidence_strength": features.get("evidence_strength"),
            },
            "feature_values": {
                "trust_band": features.get("trust_band"),
                "scam_band": features.get("scam_band"),
                "group_muted": features.get("group_muted"),
                "is_direct_mention": features.get("is_direct_mention"),
                "urgency_band": features.get("urgency_band"),
                "evidence_strength": features.get("evidence_strength"),
            },
            "conflict_count": conflict_count,
            "retrieved_evidence": [
                {
                    "message_id": item.get("message_id"),
                    "similarity": item.get("similarity"),
                    "reaction": item.get("reaction"),
                    "snippet": item.get("snippet"),
                }
                for item in evidence
            ],
            "llm_called": llm_used,
            "llm_failed": llm_used and not llm_succeeded,
            "llm_succeeded": llm_succeeded,
            "fallback_used": fallback_used,
            "top_signals": top_signals,
            "decision": decision,
            "media_summary": media_info.get("media_summary") if media_info else None,
            "extraction_source": media_info.get("extraction_source") if media_info else None,
        }
        with open(self.debug_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _evidence_ids(self, evidence: List[Dict[str, object]]) -> str:
        if not evidence:
            return "none"
        ids = [item.get("message_id") for item in evidence[:2] if item.get("message_id")]
        return ";".join(ids) if ids else "none"

    def _has_explicit_window(self, text: str) -> bool:
        return bool(re.search(r"today|tomorrow|before|by [0-9]|deadline|eod|tonight|now|immediately|until|by 5|by 6|by 7|pm|am", text.lower()))

    def _is_business_transaction(self, message: Dict[str, str], text: str, features: Dict[str, object]) -> bool:
        if not self._looks_payment(text) and not self._looks_business(text):
            return False
        return bool(message.get("business_id"))

    def _is_marketing(self, message: Dict[str, str], text: str, features: Dict[str, object]) -> bool:
        return self._looks_promo(text) and bool(message.get("business_id"))

    def _looks_personal(self, message: Dict[str, str], text: str) -> bool:
        return bool(re.search(r"@u_|can you|could you|please|collect|call me|need you|confirm", text.lower())) and not self._looks_promo(text)

    def _looks_business(self, text: str) -> bool:
        return bool(re.search(r"business|customer|account|delivery|update|reminder|review|service|appointment|schedule|booking|order|refund|wallet|bank|card|statement", text.lower()))

    def _looks_promo(self, text: str) -> bool:
        return bool(re.search(r"offer|promo|promotion|discount|sale|welcome offer|limited shopping benefit|shop|tap below|checkout|deal", text.lower()))

    def _looks_payment(self, text: str) -> bool:
        return bool(re.search(r"payment|pay|otp|verify|wallet|bank|refund|card|booking|delivery|fee|account|security", text.lower()))

    def _looks_urgent(self, text: str) -> bool:
        return bool(re.search(r"urgent|today|tomorrow|before|deadline|eod|tonight|immediately|now|expires|close at|until", text.lower()))

    def _looks_event(self, text: str) -> bool:
        return bool(re.search(r"fire alarm|field trip|exam|appointment|schedule|circular|consent|calendar|reminder|test tomorrow|tomorrow|this week|event|program|meeting", text.lower()))

    def _looks_spam(self, text: str) -> bool:
        return bool(re.search(r"win|free money|cash reward|claim now|limited time|buy now|click here|act now|subscribe|guaranteed|urgent share|forward this|blessing|health tip", text.lower())) and not self._looks_payment(text) and not self._looks_urgent(text)

    def _is_forward_pattern(self, text: str) -> bool:
        return bool(re.search(r"forward|blessing|health tip|share with|share this|luck changes|do not ignore|chain", text.lower()))

    def _final_confidence(self, action: str, message_type: str, features: Dict[str, object], evidence: List[Dict[str, object]], base_confidence: float, llm_used: bool) -> float:
        evidence_strength = features.get("evidence_strength", 0.0)
        rule_certainty = features.get("rule_certainty", 0.5)
        confidence = 0.7 * rule_certainty + 0.3 * evidence_strength
        if llm_used:
            confidence = 0.5 * 0.85 + 0.3 * evidence_strength + 0.2 * 0.8
        return round(max(0.0, min(1.0, confidence)), 3)
