"""
session/manager.py
──────────────────
Conversation session management.

Primary backend: Redis (clustered, persistent, TTL-managed)
Fallback: in-process dict (single-replica only)

Sessions store the last N conversation turns as a list of
{"role": "user"|"assistant", "content": "..."} dicts.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from core.interfaces import BaseSessionStore
from core.logger import get_logger
from config.settings import SessionConfig

log = get_logger("session.manager")


class RedisSessionStore(BaseSessionStore):
    """
    Redis-backed session store.

    Keys: {prefix}{session_id}
    Values: JSON-serialized list of message dicts
    TTL: cfg.ttl_seconds (refreshed on each write)
    """

    def __init__(self, cfg: SessionConfig) -> None:
        import redis

        self.cfg = cfg
        rc = cfg.redis
        self._client = redis.Redis(
            host=rc.host,
            port=rc.port,
            db=rc.db,
            decode_responses=True,
            socket_connect_timeout=rc.socket_timeout,
        )
        self._prefix = rc.key_prefix

        try:
            self._client.ping()
            log.info("Redis session store connected", host=rc.host, port=rc.port)
        except Exception as exc:
            raise ConnectionError(
                f"Redis unavailable at {rc.host}:{rc.port} — {exc}"
            ) from exc

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        raw = self._client.get(self._key(session_id))
        return json.loads(raw) if raw else []

    def append_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        history = self.get_history(session_id)
        history.extend([
            {"role": "user",      "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])
        # Keep only last N turns (2 messages per turn)
        max_msgs = self.cfg.max_history_turns * 2
        history = history[-max_msgs:]

        self._client.setex(
            self._key(session_id),
            self.cfg.ttl_seconds,
            json.dumps(history, ensure_ascii=False),
        )

    def clear(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))
        log.debug("Session cleared", session_id=session_id)

    def exists(self, session_id: str) -> bool:
        return bool(self._client.exists(self._key(session_id)))

    def get_ttl(self, session_id: str) -> int:
        """Return remaining TTL in seconds. -1 if no expiry, -2 if key missing."""
        return self._client.ttl(self._key(session_id))


class InMemorySessionStore(BaseSessionStore):
    """
    In-process session store.
    Use for single-replica deployments, local dev, or when Redis is unavailable.

    Warning: sessions are lost on process restart and not shared across replicas.
    """

    def __init__(self, cfg: SessionConfig) -> None:
        self.cfg = cfg
        self._store: Dict[str, List[Dict[str, str]]] = {}
        log.warning(
            "Using in-memory session store — sessions will NOT persist across restarts "
            "and are NOT shared between replicas"
        )

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._store.get(session_id, []))

    def append_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        history = self.get_history(session_id)
        history.extend([
            {"role": "user",      "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])
        max_msgs = self.cfg.max_history_turns * 2
        self._store[session_id] = history[-max_msgs:]

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def all_session_ids(self) -> List[str]:
        return list(self._store.keys())

    def session_count(self) -> int:
        return len(self._store)


def create_session_store(cfg: SessionConfig) -> BaseSessionStore:
    """
    Factory: create session store with Redis → in-memory fallback.
    """
    if cfg.backend == "memory":
        return InMemorySessionStore(cfg)

    try:
        return RedisSessionStore(cfg)
    except ConnectionError as exc:
        log.warning(
            "Redis unavailable, falling back to in-memory sessions",
            error=str(exc),
        )
        return InMemorySessionStore(cfg)
