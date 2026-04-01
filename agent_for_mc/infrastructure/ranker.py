from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_for_mc.domain.errors import ConfigurationError, ServiceError
from agent_for_mc.domain.models import RetrievedDoc
from agent_for_mc.infrastructure.observability import record_counter, trace_operation


@dataclass(slots=True)
class BceRanker:
    model_name_or_path: str
    _model: Any | None = None

    def warmup(self) -> None:
        """Load the underlying ranker model eagerly."""
        self._get_model()

    def compute_scores(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        with trace_operation(
            "ranker.compute_scores",
            attributes={"component": "ranker", "passage.count": len(passages)},
            metric_name="rag_ranker_compute_scores_seconds",
        ):
            record_counter("rag_ranker_compute_scores_requests_total")
            model = self._get_model()
            sentence_pairs = [[query, passage] for passage in passages]
            try:
                scores = model.compute_score(sentence_pairs)
            except Exception as exc:  # pragma: no cover - external model failure
                raise ServiceError(f"BCE ranker 评分失败: {exc}") from exc
            return [float(score) for score in scores]

    def rank(self, query: str, passages: list[str]) -> list[str]:
        if not passages:
            return []
        with trace_operation(
            "ranker.rank",
            attributes={"component": "ranker", "passage.count": len(passages)},
            metric_name="rag_ranker_rank_seconds",
        ):
            record_counter("rag_ranker_rank_requests_total")
            model = self._get_model()
            try:
                results = model.rerank(query, passages)
            except Exception as exc:  # pragma: no cover - external model failure
                raise ServiceError(f"BCE ranker 排序失败: {exc}") from exc

            if isinstance(results, list):
                ordered_passages: list[str] = []
                for item in results:
                    if isinstance(item, dict):
                        passage = item.get("text") or item.get("passage") or item.get("content")
                        if passage is not None:
                            ordered_passages.append(str(passage))
                            continue
                    ordered_passages.append(str(item))
                return ordered_passages
            return [str(item) for item in results]

    def rank_docs(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not docs:
            return []
        with trace_operation(
            "ranker.rank_docs",
            attributes={"component": "ranker", "doc.count": len(docs)},
            metric_name="rag_ranker_rank_docs_seconds",
        ):
            record_counter("rag_ranker_rank_docs_requests_total")
            passages = [_doc_to_passage(doc) for doc in docs]
            scores = self.compute_scores(query, passages)
            scored_docs = list(enumerate(zip(docs, scores)))
            scored_docs.sort(key=lambda item: (item[1][1], -item[0]), reverse=True)
            return [doc for _, (doc, _) in scored_docs]

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from BCEmbedding import RerankerModel
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise ConfigurationError(
                "BCEmbedding 未安装，无法启用 ranker。请先安装 BCEmbedding。"
            ) from exc

        self._model = RerankerModel(model_name_or_path=self.model_name_or_path)
        return self._model


def _doc_to_passage(doc: RetrievedDoc) -> str:
    parts = [
        f"Chinese name: {doc.plugin_chinese_name}",
        f"English name: {doc.plugin_english_name}",
        f"Content: {doc.content}",
    ]
    return "\n".join(parts)
