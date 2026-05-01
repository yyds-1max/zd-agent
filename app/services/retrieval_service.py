from __future__ import annotations

import math
import re
from collections import Counter

from app.schemas.intent import IntentResult
from app.schemas.knowledge import KnowledgeChunk, KnowledgeDocument, RetrievedChunk
from app.services.langchain_support import (
    BM25Retriever,
    LANGCHAIN_AVAILABLE,
    LangChainDocument,
    RunnableLambda,
    RunnableParallel,
)


class RetrievalService:
    def __init__(
        self,
        vector_store=None,
        rerank_service=None,
        candidate_limit: int = 8,
    ):
        self.vector_store = vector_store
        self.rerank_service = rerank_service
        self.candidate_limit = candidate_limit
        self.last_vector_backend = "local_vector_fallback"
        self.last_rerank_backend = "heuristic_rerank"

    def search(
        self,
        question: str,
        intent: IntentResult,
        documents: list[KnowledgeDocument],
        top_k: int,
        chunks: list[KnowledgeChunk] | None = None,
    ) -> list[RetrievedChunk]:
        if not documents:
            return []

        source_chunks = chunks or [self._document_as_chunk(document) for document in documents]
        if not source_chunks:
            return []

        if LANGCHAIN_AVAILABLE:
            try:
                return self._search_with_langchain(
                    question=question,
                    intent=intent,
                    chunks=source_chunks,
                    top_k=top_k,
                )
            except Exception:
                pass

        query_tokens = self._tokenize(
            " ".join([question, *intent.keywords, *intent.project_names])
        )
        bm25_scores = self._bm25_scores(source_chunks, query_tokens)
        vector_scores = self._search_vector_scores(question, source_chunks, query_tokens, top_k)
        bm25_rank = self._ranking_from_scores(bm25_scores)
        vector_rank = self._ranking_from_scores(vector_scores)
        candidate_ids = self._merge_candidates(bm25_rank, vector_rank, top_k)
        rerank_scores = self._rerank_scores(question, intent, source_chunks, candidate_ids)
        return self._build_retrieved_chunks(
            chunks=source_chunks,
            query_tokens=query_tokens,
            bm25_scores=bm25_scores,
            vector_scores=vector_scores,
            rerank_scores=rerank_scores,
            candidate_ids=candidate_ids,
            bm25_rank=bm25_rank,
            vector_rank=vector_rank,
            top_k=top_k,
        )

    def describe_backend(self) -> str:
        return f"{self.last_vector_backend} + {self.last_rerank_backend}"

    def _search_with_langchain(
        self,
        *,
        question: str,
        intent: IntentResult,
        chunks: list[KnowledgeChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        lc_chunks = [self._to_langchain_chunk(chunk) for chunk in chunks]
        bm25_k = max(top_k * 3, self.candidate_limit)
        bm25_retriever = BM25Retriever.from_documents(
            lc_chunks,
            preprocess_func=self._tokenize,
            k=bm25_k,
        )
        hybrid_chain = RunnableParallel(
            bm25=RunnableLambda(lambda payload: bm25_retriever.invoke(payload["question"])),
            vector=RunnableLambda(
                lambda payload: self._langchain_vector_search(
                    question=payload["question"],
                    chunks=payload["chunks"],
                    top_k=payload["top_k"],
                )
            ),
        )
        results = hybrid_chain.invoke(
            {
                "question": question,
                "chunks": chunks,
                "top_k": top_k,
            }
        )

        bm25_chunks = results["bm25"]
        vector_scores = results["vector"]
        bm25_rank = {
            item.metadata["chunk_id"]: index + 1
            for index, item in enumerate(bm25_chunks)
        }
        bm25_scores = {
            item.metadata["chunk_id"]: max(0.0, 1 - (index / max(len(bm25_chunks), 1)))
            for index, item in enumerate(bm25_chunks)
        }
        vector_rank = self._ranking_from_scores(vector_scores)
        candidate_ids = self._merge_candidates(bm25_rank, vector_rank, top_k)
        rerank_scores = self._rerank_scores(question, intent, chunks, candidate_ids)
        return self._build_retrieved_chunks(
            chunks=chunks,
            query_tokens=self._tokenize(question),
            bm25_scores=bm25_scores,
            vector_scores=vector_scores,
            rerank_scores=rerank_scores,
            candidate_ids=candidate_ids,
            bm25_rank=bm25_rank,
            vector_rank=vector_rank,
            top_k=top_k,
        )

    def _build_retrieved_chunks(
        self,
        *,
        chunks: list[KnowledgeChunk],
        query_tokens: list[str],
        bm25_scores: dict[str, float],
        vector_scores: dict[str, float],
        rerank_scores: dict[str, float],
        candidate_ids: set[str],
        bm25_rank: dict[str, int],
        vector_rank: dict[str, int],
        top_k: int,
    ) -> list[RetrievedChunk]:
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        bm25_max = max(bm25_scores.values(), default=1.0) or 1.0
        vector_max = max(vector_scores.values(), default=1.0) or 1.0

        retrieved: list[RetrievedChunk] = []
        for chunk_id in candidate_ids:
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue
            normalized_bm25 = bm25_scores.get(chunk_id, 0.0) / bm25_max
            normalized_vector = vector_scores.get(chunk_id, 0.0) / vector_max
            rrf_score = self._rrf_score(bm25_rank.get(chunk_id), vector_rank.get(chunk_id))
            rerank_score = rerank_scores.get(chunk_id, 0.0)
            final_score = (
                0.30 * normalized_bm25
                + 0.20 * normalized_vector
                + 18 * rrf_score
                + 0.50 * rerank_score
            )
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    snippet=self._chunk_snippet(chunk, query_tokens),
                    bm25_score=normalized_bm25,
                    vector_score=normalized_vector,
                    rrf_score=rrf_score,
                    rerank_score=rerank_score,
                    final_score=final_score,
                )
            )

        retrieved.sort(key=lambda item: item.final_score, reverse=True)
        return retrieved[:top_k]

    def _bm25_scores(
        self, chunks: list[KnowledgeChunk], query_tokens: list[str]
    ) -> dict[str, float]:
        tokenized_chunks = {chunk.chunk_id: self._tokenize(chunk.searchable_text()) for chunk in chunks}
        avg_doc_len = sum(len(tokens) for tokens in tokenized_chunks.values()) / max(len(chunks), 1)
        doc_freq = Counter()
        for tokens in tokenized_chunks.values():
            for token in set(tokens):
                doc_freq[token] += 1

        k1 = 1.5
        b = 0.75
        scores: dict[str, float] = {}
        for chunk in chunks:
            tokens = tokenized_chunks[chunk.chunk_id]
            term_freq = Counter(tokens)
            doc_length = max(len(tokens), 1)
            score = 0.0
            for token in query_tokens:
                if term_freq[token] == 0:
                    continue
                idf = math.log(
                    1 + (len(chunks) - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5)
                )
                numerator = term_freq[token] * (k1 + 1)
                denominator = term_freq[token] + k1 * (
                    1 - b + b * (doc_length / max(avg_doc_len, 1.0))
                )
                score += idf * numerator / denominator
            scores[chunk.chunk_id] = score
        return scores

    def _search_vector_scores(
        self,
        question: str,
        chunks: list[KnowledgeChunk],
        query_tokens: list[str],
        top_k: int,
    ) -> dict[str, float]:
        if self.vector_store is not None:
            scores = self.vector_store.search_chunk_scores(
                query_text=question,
                allowed_doc_ids=[chunk.doc_id for chunk in chunks],
                n_results=max(top_k * 3, self.candidate_limit),
            )
            if scores:
                self.last_vector_backend = self.vector_store.describe()
                return scores
        self.last_vector_backend = "local_vector_fallback"
        return self._local_vector_scores(chunks, query_tokens)

    def _langchain_vector_search(
        self,
        *,
        question: str,
        chunks: list[KnowledgeChunk],
        top_k: int,
    ) -> dict[str, float]:
        if self.vector_store is not None:
            scores = self.vector_store.search_chunk_scores(
                query_text=question,
                allowed_doc_ids=[chunk.doc_id for chunk in chunks],
                n_results=max(top_k * 3, self.candidate_limit),
            )
            if scores:
                self.last_vector_backend = self.vector_store.describe()
                return scores
        self.last_vector_backend = "local_vector_fallback"
        return self._local_vector_scores(chunks, self._tokenize(question))

    def _local_vector_scores(
        self, chunks: list[KnowledgeChunk], query_tokens: list[str]
    ) -> dict[str, float]:
        tokenized_chunks = {chunk.chunk_id: self._tokenize(chunk.searchable_text()) for chunk in chunks}
        doc_freq = Counter()
        for tokens in tokenized_chunks.values():
            for token in set(tokens):
                doc_freq[token] += 1

        query_weights = self._tfidf_weights(query_tokens, doc_freq, len(chunks))
        query_norm = math.sqrt(sum(value * value for value in query_weights.values())) or 1.0

        scores: dict[str, float] = {}
        for chunk in chunks:
            chunk_weights = self._tfidf_weights(
                tokenized_chunks[chunk.chunk_id], doc_freq, len(chunks)
            )
            chunk_norm = math.sqrt(sum(value * value for value in chunk_weights.values())) or 1.0
            dot = sum(
                query_weights.get(token, 0.0) * chunk_weights.get(token, 0.0)
                for token in query_weights
            )
            scores[chunk.chunk_id] = dot / (query_norm * chunk_norm)
        return scores

    def _tfidf_weights(
        self, tokens: list[str], doc_freq: Counter[str], total_docs: int
    ) -> dict[str, float]:
        term_freq = Counter(tokens)
        weights: dict[str, float] = {}
        for token, tf in term_freq.items():
            idf = math.log(1 + total_docs / (1 + doc_freq[token]))
            weights[token] = tf * idf
        return weights

    def _ranking_from_scores(self, scores: dict[str, float]) -> dict[str, int]:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {item_id: index + 1 for index, (item_id, _) in enumerate(ordered)}

    def _merge_candidates(
        self,
        bm25_rank: dict[str, int],
        vector_rank: dict[str, int],
        top_k: int,
    ) -> set[str]:
        limit = max(top_k * 3, self.candidate_limit)
        candidates = {item_id for item_id, rank in bm25_rank.items() if rank <= limit}
        candidates.update(item_id for item_id, rank in vector_rank.items() if rank <= limit)
        return candidates

    def _rrf_score(self, bm25_rank: int | None, vector_rank: int | None) -> float:
        score = 0.0
        for rank in [bm25_rank, vector_rank]:
            if rank is not None:
                score += 1 / (60 + rank)
        return score

    def _rerank_scores(
        self,
        question: str,
        intent: IntentResult,
        chunks: list[KnowledgeChunk],
        candidate_ids: set[str],
    ) -> dict[str, float]:
        candidate_chunks = [chunk for chunk in chunks if chunk.chunk_id in candidate_ids]
        if self.rerank_service is not None and self.rerank_service.is_available():
            try:
                scores = self.rerank_service.rerank(question, candidate_chunks)
                self.last_rerank_backend = "dashscope:gte-rerank-v2"
                return scores
            except Exception:
                pass

        self.last_rerank_backend = "heuristic_rerank"
        return {
            chunk.chunk_id: self._heuristic_rerank(question, intent, chunk)
            for chunk in candidate_chunks
        }

    def _heuristic_rerank(
        self,
        question: str,
        intent: IntentResult,
        chunk: KnowledgeChunk,
    ) -> float:
        score = 0.0
        if chunk.is_latest:
            score += 0.4
        if intent.name == "project_lookup" and chunk.doc_type in {
            "project_requirement",
            "project_weekly",
            "project_plan",
            "chat_summary",
        }:
            score += 1.2
        if intent.name == "policy_lookup" and chunk.doc_type in {
            "policy",
            "finance_rule",
            "faq",
        }:
            score += 1.1
        if intent.name == "faq_lookup" and chunk.doc_type == "faq":
            score += 1.0
        for project_name in intent.project_names:
            if project_name and project_name in chunk.doc_title:
                score += 1.3
        searchable_text = chunk.searchable_text().lower()
        for keyword in intent.keywords:
            if keyword and keyword in chunk.doc_title.lower():
                score += 0.6
            if keyword and keyword in searchable_text:
                score += 0.2
        if "最新" in question and chunk.is_latest:
            score += 0.4
        return min(score / 4.0, 1.0)

    def _chunk_snippet(self, chunk: KnowledgeChunk, query_tokens: list[str]) -> str:
        lines = [line for line in chunk.text.splitlines() if line]
        if not lines:
            return chunk.text[:220]
        scored: list[tuple[int, str]] = []
        for line in lines:
            tokens = self._tokenize(line)
            overlap = len(set(tokens) & set(query_tokens))
            scored.append((overlap, line))
        scored.sort(key=lambda item: item[0], reverse=True)
        return (scored[0][1] if scored else chunk.text)[:220]

    def _to_langchain_chunk(self, chunk: KnowledgeChunk):
        return LangChainDocument(
            page_content=chunk.searchable_text(),
            metadata=chunk.to_metadata(),
        )

    def _document_as_chunk(self, document: KnowledgeDocument) -> KnowledgeChunk:
        return KnowledgeChunk(
            chunk_id=f"{document.doc_id}::chunk::000",
            doc_id=document.doc_id,
            chunk_index=0,
            doc_title=document.title,
            doc_type=document.doc_type,
            topic=document.topic,
            permission_level=document.permission_level,
            version=document.version,
            status=document.status,
            published_at=document.published_at,
            updated_at=document.updated_at,
            is_latest=document.is_latest,
            project_name=document.project_name,
            source_path=document.source_path,
            section_title=None,
            subsection_title=None,
            text=document.body,
        )

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for chunk in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
            if re.fullmatch(r"[A-Za-z0-9]+", chunk):
                tokens.append(chunk)
                continue
            tokens.append(chunk)
            if len(chunk) <= 2:
                continue
            for index in range(len(chunk) - 1):
                tokens.append(chunk[index : index + 2])
        return tokens
