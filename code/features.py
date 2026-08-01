import re
from typing import Dict, List, Optional


def _safe_int(value: object, default: int = 0) -> int:
    """int() that tolerates missing keys, None, and blank/whitespace strings from CSV data."""
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except (ValueError, TypeError):
        return default


class FeatureEngine:
    def __init__(self, users: List[Dict[str, str]], groups: List[Dict[str, str]], group_members: List[Dict[str, str]], business_accounts: List[Dict[str, str]], user_business_history: List[Dict[str, str]], message_events: List[Dict[str, str]], daily_notification_summary: Optional[List[Dict[str, str]]] = None):
        self.users = {row.get("user_id"): row for row in users}
        self.groups = {row.get("group_id"): row for row in groups}
        self.group_members = {(row.get("group_id"), row.get("user_id")): row for row in group_members}
        self.business_accounts = {row.get("business_id"): row for row in business_accounts}
        self.user_business_history = {(row.get("user_id"), row.get("business_id")): row for row in user_business_history}
        self.message_events = {(row.get("user_id"), row.get("message_id")): row for row in message_events}
        self.fatigue_by_user = self._build_fatigue_index(daily_notification_summary or [])

    def _build_fatigue_index(self, daily_summary: List[Dict[str, str]]) -> Dict[str, float]:
        """Per-user notification dismissal rate from daily_notification_summary.csv.
        A high recent dismissal rate is a personalization signal: this user tends to
        ignore notifications, so borderline messages should lean toward digest rather
        than notify."""
        sent_by_user: Dict[str, int] = {}
        dismissed_by_user: Dict[str, int] = {}
        for row in daily_summary:
            user_id = row.get("user_id")
            if not user_id:
                continue
            sent_by_user[user_id] = sent_by_user.get(user_id, 0) + _safe_int(row.get("notifications_sent"))
            dismissed_by_user[user_id] = dismissed_by_user.get(user_id, 0) + _safe_int(row.get("notifications_dismissed"))
        fatigue: Dict[str, float] = {}
        for user_id, sent in sent_by_user.items():
            if sent <= 0:
                fatigue[user_id] = 0.0
                continue
            rate = dismissed_by_user.get(user_id, 0) / sent
            fatigue[user_id] = round(max(0.0, min(1.0, rate)), 3)
        return fatigue

    def build_features(self, message: Dict[str, str], evidence: List[Dict[str, object]], media_info: Dict[str, object]) -> Dict[str, object]:
        user = self.users.get(message.get("user_id"), {})
        group_member = self.group_members.get((message.get("group_id"), message.get("user_id")), {}) if message.get("group_id") else {}
        business = self.business_accounts.get(message.get("business_id"), {}) if message.get("business_id") else {}
        business_history = self.user_business_history.get((message.get("user_id"), message.get("business_id")), {}) if message.get("business_id") else {}

        trust_score = self._trust_score(user, group_member, business, business_history)
        business_repeat_dismissed = self._business_repeat_dismissed(business, business_history)
        urgency_score = self._urgency_score(message, media_info)
        scam_score = self._scam_score(message, business, media_info, evidence)
        notification_fatigue = self.fatigue_by_user.get(message.get("user_id"), 0.0)
        untranscribed_voice = bool(message.get("media_type") == "voice" and media_info.get("extraction_source") != "asr")
        return {
            "trust_score": trust_score,
            "trust_band": self._band(trust_score, "trust"),
            "urgency_score": urgency_score,
            "urgency_band": self._band(urgency_score, "urgency"),
            "scam_score": scam_score,
            "scam_band": self._band(scam_score, "scam"),
            "role": group_member.get("role") or "",
            "group_muted": bool(group_member.get("group_muted_by_user") == "1"),
            "group_type": self.groups.get(message.get("group_id"), {}).get("group_type") if message.get("group_id") else "",
            "business_verified": bool(business.get("verified") == "1"),
            "allows_promotions": bool(business_history.get("allows_promotions") == "1") if business_history else False,
            "activity_count_180d": _safe_int(business_history.get("activity_count_180d") if business_history else None),
            "is_duplicate": self._is_duplicate(message, evidence),
            "is_direct_mention": bool(re.search(r"@u_\d+|@\w+", (message.get("message_text") or ""))),
            "evidence_strength": self._evidence_strength(evidence),
            "rule_certainty": self._rule_certainty(urgency_score, scam_score, trust_score),
            "notification_fatigue": notification_fatigue,
            "high_fatigue": notification_fatigue >= 0.6,
            "untranscribed_voice": untranscribed_voice,
            "business_repeat_dismissed": business_repeat_dismissed,
        }

    def _business_repeat_dismissed(self, business: Dict[str, str], business_history: Dict[str, str]) -> bool:
        """An unverified, heavily-reported business that this user has repeatedly
        dismissed with zero opens or replies is a strong spam signal that the
        keyword-based content classifier can't see on its own. Requires all of:
        unverified sender, several dismissals with no engagement, and a
        meaningfully high user-report count on the business itself - each signal
        alone is too weak (verified businesses get dismissed too; new users have
        low report counts by default)."""
        if not business_history:
            return False
        if business.get("verified") == "1":
            return False
        dismissed = _safe_int(business_history.get("messages_dismissed_30d"))
        replied = _safe_int(business_history.get("messages_replied_30d"))
        opened = _safe_int(business_history.get("messages_opened_30d"))
        reports = _safe_int(business.get("user_reports_30d"))
        return dismissed >= 3 and replied == 0 and opened == 0 and reports >= 10

    def _trust_score(self, user: Dict[str, str], group_member: Dict[str, str], business: Dict[str, str], business_history: Dict[str, str]) -> float:
        weights = []
        values = []
        if business_history and "activity_count_180d" in business_history:
            # Past activity alone isn't positive trust signal - a user who has
            # dismissed every message and never opened or replied has a *negative*
            # relationship with this sender, not a neutral/positive one. Only count
            # activity as trust-building when it reflects real engagement (a reply,
            # or opens outweighing dismissals).
            activity = int(business_history.get("activity_count_180d", "0"))
            dismissed = _safe_int(business_history.get("messages_dismissed_30d"))
            replied = _safe_int(business_history.get("messages_replied_30d"))
            opened = _safe_int(business_history.get("messages_opened_30d"))
            if activity <= 0:
                relationship = 0.0
            elif replied > 0:
                relationship = 1.0
            elif opened > 0 and dismissed <= opened:
                relationship = 0.8
            elif dismissed >= 3 and opened == 0 and replied == 0:
                relationship = 0.0
            else:
                relationship = 0.5
            weights.append(0.35)
            values.append(relationship)
        if business and "verified" in business:
            verified = 1.0 if business.get("verified") == "1" else 0.0
            weights.append(0.25)
            values.append(verified)
        if group_member and "role" in group_member:
            role = 1.0 if group_member.get("role") in {"admin", "owner"} else 0.0
            weights.append(0.15)
            values.append(role)
        if user and "messages_opened_30d" in user:
            age_factor = min(1.0, int(user.get("messages_opened_30d", "0")) / 60.0)
            weights.append(0.15)
            values.append(age_factor)
        if user and "messages_reported_30d" in user:
            report_factor = max(0.0, 1.0 - min(1.0, int(user.get("messages_reported_30d", "0")) / 10.0))
            weights.append(-0.10)
            values.append(report_factor)
        if not weights:
            return 0.5
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5
        score = sum(w * v for w, v in zip(weights, values)) / total_weight
        return round(max(0.0, min(1.0, score)), 3)

    def _urgency_score(self, message: Dict[str, str], media_info: Dict[str, object]) -> float:
        text = (message.get("message_text") or "").lower()
        summary = (media_info.get("media_summary") or "").lower()
        combined = f"{text} {summary}"
        score = 0.0
        if re.search(r"today|tomorrow|before|by [0-9]|deadline|eod|tonight|now|immediately|urgent|expires|close at|until", combined):
            score += 0.30
        if re.search(r"emergency|safety|risk|lock|block|account|security|urgent|medical|clinic|doctor|hospital", combined):
            score += 0.20
        if re.search(r"admin|notice|reminder|system note|maintenance|update", combined):
            score += 0.20
        if re.search(r"payment|otp|verify|delivery|booking|refund|wallet|card|pay|review|submit|reply", combined):
            score += 0.15
        if _safe_int(message.get("forwarded_count")) > 0:
            score -= 0.20
        if re.search(r"offer|promotion|sale|discount|free|limited|welcome|blessing|health tip|share", combined):
            score -= 0.15
        return round(max(0.0, min(1.0, score)), 3)

    def _scam_score(self, message: Dict[str, str], business: Dict[str, str], media_info: Dict[str, object], evidence: List[Dict[str, object]]) -> float:
        text = (message.get("message_text") or "").lower()
        combined = f"{text} {(media_info.get('media_summary') or '').lower()}"
        score = 0.0
        if not business or business.get("verified") != "1":
            score += 0.35
        if re.search(r"otp|verify|payment|pay|wallet|bank|refund|code|account|security|link|scan|click", combined):
            score += 0.25
        if re.search(r"bit\.ly|http|https|verify-quick|amazonpay|delivery|link|qr", combined):
            score += 0.20
        if _safe_int(message.get("forwarded_count")) > 0:
            score += 0.15
        if re.search(r"imperson|unknown|is this|from the courier desk|helping you|account block|security at risk", combined):
            score += 0.10
        if re.search(r"urgent|today|now|immediately|reply|code", combined) and score > 0:
            score += 0.05
        return round(max(0.0, min(1.0, score)), 3)

    def _band(self, value: float, feature: str) -> str:
        if feature == "scam":
            if value >= 0.88:
                return "High"
            if value >= 0.60:
                return "Medium"
            return "Low"
        if feature == "urgency":
            if value >= 0.65:
                return "High"
            if value >= 0.30:
                return "Medium"
            return "Low"
        if feature == "trust":
            if value >= 0.70:
                return "High"
            if value >= 0.35:
                return "Medium"
            return "Low"
        return "Low"

    def _is_duplicate(self, message: Dict[str, str], evidence: List[Dict[str, object]]) -> bool:
        return bool(evidence and evidence[0].get("similarity", 0.0) > 0.85)

    def _evidence_strength(self, evidence: List[Dict[str, object]]) -> float:
        if not evidence:
            return 0.0
        top = evidence[0]
        same_sender = 1.0 if top.get("same_sender") else 0.0
        consistent = 1.0 if top.get("reaction") in {"opened_replied", "opened", "replied"} else 0.0
        return round(0.6 * float(top.get("similarity", 0.0)) + 0.2 * same_sender + 0.2 * consistent, 3)

    def _rule_certainty(self, urgency_score: float, scam_score: float, trust_score: float) -> float:
        certainty = 0.5 + 0.2 * urgency_score + 0.2 * (1.0 - scam_score) + 0.1 * trust_score
        return round(max(0.0, min(1.0, certainty)), 3)
