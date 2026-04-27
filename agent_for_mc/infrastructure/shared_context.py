from __future__ import annotations

from threading import Lock
from typing import Generic, TypeVar, cast


T = TypeVar("T")


class SharedContextSlot(Generic[T]):
    """Thread-safe process-wide slot for long-lived tool dependencies."""

    def __init__(self, name: str):
        self._name = name
        self._lock = Lock()
        self._value: T | None = None

    def set(self, value: T) -> None:
        with self._lock:
            self._value = value

    def get(self, *, error_message: str) -> T:
        with self._lock:
            value = self._value
        if value is None:
            raise RuntimeError(error_message)
        return cast(T, value)
