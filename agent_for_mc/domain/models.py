from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RetrievedDoc:
    id: int
    plugin_chinese_name: str
    plugin_english_name: str
    content: str
    distance: float
    match_reason: str = "vector"


@dataclass(slots=True)
class PluginConfigDoc:
    id: int
    plugin_chinese_name: str
    plugin_english_name: str
    file_path: str
    content: str
    distance: float
    match_reason: str = "vector"


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[RetrievedDoc] = field(default_factory=list)
    standalone_query: str = ""

    @property
    def rewritten_question(self) -> str:
        return self.standalone_query

    @rewritten_question.setter
    def rewritten_question(self, value: str) -> None:
        self.standalone_query = value


@dataclass(slots=True)
class VectorStoreStats:
    db_dir: Path
    table_name: str
    record_count: int
    embedding_dimension: int
