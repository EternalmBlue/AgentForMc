from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

import grpc

from agent_for_mc.domain.errors import ServiceError
from agent_for_mc.domain.models import RetrievedDoc
from agent_for_mc.interfaces.grpc import reranker_pb2, reranker_pb2_grpc


class Ranker(Protocol):
    def rank_docs(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        ...


@dataclass(slots=True)
class GrpcRerankerClient:
    host: str
    port: int
    auth_token: str
    timeout_seconds: float = 10.0
    _channel: grpc.Channel | None = field(default=None, init=False, repr=False)
    _stub: reranker_pb2_grpc.RerankerServiceStub | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def rank_docs(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not docs:
            return []

        request = reranker_pb2.RerankRequest(
            request_id=str(uuid4()),
            query=query,
            documents=[
                reranker_pb2.RerankDocument(
                    index=index,
                    document_id=str(doc.id),
                    text=_doc_to_passage(doc),
                )
                for index, doc in enumerate(docs)
            ],
        )
        try:
            response = self._get_stub().Rerank(
                request,
                metadata=self._metadata(),
                timeout=self.timeout_seconds,
            )
        except grpc.RpcError as exc:
            detail = exc.details() if hasattr(exc, "details") else str(exc)
            raise ServiceError(f"reranker gRPC request failed: {detail}") from exc
        except Exception as exc:  # pragma: no cover - defensive transport wrapper
            raise ServiceError(f"reranker gRPC request failed: {exc}") from exc

        return _apply_ranked_indexes(docs, [item.index for item in response.results])

    def health(self) -> reranker_pb2.HealthResponse:
        try:
            return self._get_stub().Health(
                reranker_pb2.HealthRequest(),
                timeout=self.timeout_seconds,
            )
        except grpc.RpcError as exc:
            detail = exc.details() if hasattr(exc, "details") else str(exc)
            raise ServiceError(f"reranker health check failed: {detail}") from exc

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def _get_stub(self) -> reranker_pb2_grpc.RerankerServiceStub:
        if self._stub is None:
            target = f"{self.host}:{self.port}"
            self._channel = grpc.insecure_channel(target)
            self._stub = reranker_pb2_grpc.RerankerServiceStub(self._channel)
        return self._stub

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        return (("authorization", f"Bearer {self.auth_token.strip()}"),)


def build_reranker_client(settings) -> GrpcRerankerClient | None:
    if not settings.reranker_enabled:
        return None
    return GrpcRerankerClient(
        host=settings.reranker_host,
        port=settings.reranker_port,
        auth_token=settings.reranker_auth_token or "",
        timeout_seconds=settings.reranker_timeout_seconds,
    )


def _apply_ranked_indexes(docs: list[RetrievedDoc], ranked_indexes: list[int]) -> list[RetrievedDoc]:
    docs_by_index = {index: doc for index, doc in enumerate(docs)}
    ordered_docs: list[RetrievedDoc] = []
    seen: set[int] = set()

    for index in ranked_indexes:
        if index in seen:
            continue
        doc = docs_by_index.get(index)
        if doc is None:
            continue
        ordered_docs.append(doc)
        seen.add(index)

    for index, doc in enumerate(docs):
        if index not in seen:
            ordered_docs.append(doc)
    return ordered_docs


def _doc_to_passage(doc: RetrievedDoc) -> str:
    parts = [
        f"Chinese name: {doc.plugin_chinese_name}",
        f"English name: {doc.plugin_english_name}",
        f"Content: {doc.content}",
    ]
    return "\n".join(parts)
