import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional


class _TFIDFBackend:
    def __init__(self) -> None:
        self.vocabulary = {}
        self.idf = {}

    def fit(self, texts: List[str]) -> None:
        docs = [self._tokenize(text) for text in texts if text]
        if not docs:
            self.vocabulary = {}
            self.idf = {}
            return
        vocab = set()
        for doc in docs:
            vocab.update(doc.keys())
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(vocab))}
        doc_freq = Counter()
        for doc in docs:
            doc_freq.update(set(doc.keys()))
        total_docs = max(1, len(docs))
        self.idf = {term: math.log((1 + total_docs) / (1 + doc_freq[term])) + 1.0 for term in self.vocabulary}

    def embed(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vector = [0.0] * len(self.vocabulary)
        for term, count in tokens.items():
            if term in self.vocabulary:
                vector[self.vocabulary[term]] = count * self.idf.get(term, 1.0)
        return vector

    def similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return round(dot / (na * nb), 3)

    def _tokenize(self, text: str) -> Counter:
        return Counter(re.findall(r"[a-zA-Z]{2,}", (text or "").lower()))


class Retriever:
    def __init__(self, message_history_rows: List[Dict[str, str]], message_events_rows: List[Dict[str, str]], similarity_floor: float = 0.55):
        self.history = message_history_rows
        self.events = message_events_rows
        self.similarity_floor = similarity_floor
        self.event_map = {(row.get("user_id"), row.get("message_id")): row for row in self.events}
        self.embedding_backend = self._initialize_embedding_backend()
        self.history_texts = [self._text_for(row) for row in self.history]
        self.history_embeddings = self._embed_many(self.history_texts)

    def retrieve(self, current_message: Dict[str, str], top_k: int = 2) -> List[Dict[str, object]]:
        current_text = self._text_for(current_message)
        current_embedding = self._embed_one(current_text)
        candidates = []
        for idx, row in enumerate(self.history):
            if row.get("user_id") != current_message.get("user_id"):
                continue
            text = (row.get("message_text") or "").strip()
            if not text:
                continue
            semantic_similarity = self._semantic_similarity(current_embedding, self.history_embeddings[idx])
            if semantic_similarity < self.similarity_floor:
                continue
            same_sender_or_business = self._same_sender_or_business(current_message, row)
            same_group_or_conversation = self._same_group_or_conversation(current_message, row)
            recency_weight = self._recency_weight(current_message.get("created_at"), row.get("created_at"))
            score = 0.5 * semantic_similarity + 0.2 * (1.0 if same_sender_or_business else 0.0) + 0.2 * (1.0 if same_group_or_conversation else 0.0) + 0.1 * recency_weight
            sim = round(min(1.0, max(0.0, score)), 3)
            if sim < self.similarity_floor:
                continue
            event = self.event_map.get((current_message.get("user_id"), row.get("message_id")), {})
            candidates.append({
                "message_id": row.get("message_id"),
                "snippet": text[:160],
                "similarity": sim,
                "same_sender": self._same_sender(current_message, row),
                "same_business": self._same_business(current_message, row),
                "same_group": self._same_group(current_message, row),
                "same_group_or_conversation": same_group_or_conversation,
                "reaction": self._reaction_summary(event),
                "created_at": row.get("created_at"),
                "event": event,
            })
        candidates.sort(key=lambda item: item["similarity"], reverse=True)
        return candidates[:top_k]

    def _initialize_embedding_backend(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            SentenceTransformer = None

        backend = None
        if SentenceTransformer is not None:
            try:
                model_name = os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2"
                backend = _SentenceTransformerBackend(model_name)
            except Exception:
                # Fall back to the provider path or the deterministic TF-IDF backend if the transformer model fails to initialize.
                backend = None

        if backend is None:
            api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key and (os.getenv("EMBEDDING_BASE_URL") or os.getenv("EMBEDDING_API_URL")):
                try:
                    backend = _ProviderEmbeddingBackend()
                except Exception:
                    # Fall back to the deterministic TF-IDF backend if the provider embedding path is unavailable.
                    backend = None

        if backend is None:
            return _TFIDFBackend()
        return backend

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if isinstance(self.embedding_backend, _TFIDFBackend):
            self.embedding_backend.fit(texts)
            return [self.embedding_backend.embed(text) for text in texts]
        return self.embedding_backend.embed_many(texts)

    def _embed_one(self, text: str) -> List[float]:
        if not text:
            return []
        if isinstance(self.embedding_backend, _TFIDFBackend):
            return self.embedding_backend.embed(text)
        return self.embedding_backend.embed(text)

    def _semantic_similarity(self, current_embedding: List[float], candidate_embedding: List[float]) -> float:
        if not current_embedding or not candidate_embedding:
            return 0.0
        if isinstance(self.embedding_backend, _TFIDFBackend):
            return self.embedding_backend.similarity(current_embedding, candidate_embedding)
        return round(min(1.0, max(0.0, self.embedding_backend.similarity(current_embedding, candidate_embedding))), 3)

    def _same_sender(self, current_message: Dict[str, str], candidate: Dict[str, str]) -> bool:
        return bool(current_message.get("sender_user_id") and candidate.get("sender_user_id") and current_message.get("sender_user_id") == candidate.get("sender_user_id"))

    def _same_business(self, current_message: Dict[str, str], candidate: Dict[str, str]) -> bool:
        return bool(current_message.get("business_id") and candidate.get("business_id") and current_message.get("business_id") == candidate.get("business_id"))

    def _same_sender_or_business(self, current_message: Dict[str, str], candidate: Dict[str, str]) -> bool:
        return self._same_sender(current_message, candidate) or self._same_business(current_message, candidate)

    def _same_group(self, current_message: Dict[str, str], candidate: Dict[str, str]) -> bool:
        return bool(current_message.get("group_id") and candidate.get("group_id") and current_message.get("group_id") == candidate.get("group_id"))

    def _same_group_or_conversation(self, current_message: Dict[str, str], candidate: Dict[str, str]) -> bool:
        same_group = self._same_group(current_message, candidate)
        same_conversation = bool(current_message.get("conversation_type") and candidate.get("conversation_type") and current_message.get("conversation_type") == candidate.get("conversation_type"))
        return same_group or same_conversation

    def _recency_weight(self, current_time: Optional[str], candidate_time: Optional[str]) -> float:
        if not current_time or not candidate_time:
            return 0.5
        try:
            from datetime import datetime
            a = datetime.strptime(current_time, "%Y-%m-%d %H:%M")
            b = datetime.strptime(candidate_time, "%Y-%m-%d %H:%M")
            delta = abs((a - b).total_seconds()) / 3600.0
            if delta <= 24:
                return 1.0
            if delta <= 72:
                return 0.7
            return 0.3
        except Exception:
            return 0.5

    def _text_for(self, message: Dict[str, str]) -> str:
        text = (message.get("message_text") or "").strip()
        if not text:
            text = (message.get("unified_text") or "")
        return re.sub(r"\s+", " ", text).lower()

    def _reaction_summary(self, event: Dict[str, str]) -> str:
        if not event:
            return "none"
        opened = event.get("message_opened") == "1"
        replied = event.get("message_replied") == "1"
        dismissed = event.get("notification_dismissed") == "1"
        reported = event.get("message_reported") == "1"
        if reported:
            return "reported"
        if opened and replied:
            return "opened_replied"
        if opened:
            return "opened"
        if dismissed:
            return "dismissed"
        if replied:
            return "replied"
        return "seen"


class _SentenceTransformerBackend:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(model_name)

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [list(item) for item in self.model.encode(texts, convert_to_numpy=True)]

    def embed(self, text: str) -> List[float]:
        return list(self.model.encode([text], convert_to_numpy=True)[0])

    def similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return round(dot / (na * nb), 3)


class _ProviderEmbeddingBackend:
    def __init__(self) -> None:
        import requests
        self.requests = requests
        self.base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("EMBEDDING_API_URL")
        self.api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = os.getenv("EMBEDDING_MODEL") or "text-embedding-004"

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        payload = {"input": texts, "model": self.model}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = self.requests.post(self.base_url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return [[float(v) for v in item.get("embedding", [])] for item in data.get("data", [])]

    def embed(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    def similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return round(dot / (na * nb), 3)
