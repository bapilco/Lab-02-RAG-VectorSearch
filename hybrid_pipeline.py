"""Extensión Parte 3.B: recuperación híbrida BM25 + densa, fusionada con RRF.

Mantiene el corpus, modelo de embeddings, Qdrant y prompt del baseline. Solo cambia el
orden de los fragmentos: top-20 denso + top-20 BM25 -> RRF -> top-k.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ingestion import Chunk
from rag_pipeline import RagPipeline


def tokenizar_bm25(texto: str) -> list[str]:
    """Tokenización léxica simple y reproducible para BM25."""
    return re.findall(r"\w+", texto.casefold(), flags=re.UNICODE)


def fusion_rrf(rankings: list[list[str]], k_rrf: int = 60) -> dict[str, float]:
    """Fusiona rankings por Reciprocal Rank Fusion, sin mezclar escalas de puntajes."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for posicion, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + posicion)
    return scores


class HybridRagPipeline(RagPipeline):
    """RAG baseline cuya recuperación se fusiona con BM25 mediante RRF."""

    def __init__(self, dense_candidates: int = 20, bm25_candidates: int = 20,
                 rrf_k: int = 60):
        super().__init__()
        if dense_candidates < 1 or bm25_candidates < 1 or rrf_k < 1:
            raise ValueError("dense_candidates, bm25_candidates y rrf_k deben ser positivos")
        self.dense_candidates = dense_candidates
        self.bm25_candidates = bm25_candidates
        self.rrf_k = rrf_k
        self._chunks: list[Chunk] = []
        self._chunk_by_id: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> None:
        """Indexa en Qdrant y construye BM25 sobre exactamente los mismos chunks."""
        super().index(chunks)
        self._chunks = chunks
        self._chunk_by_id = {chunk.id: chunk for chunk in chunks}
        self._bm25 = BM25Okapi([tokenizar_bm25(chunk.text) for chunk in chunks])

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        if self._bm25 is None:
            raise RuntimeError("Primero ejecuta index(chunks): BM25 se construye sobre los chunks indexados")

        dense_hits = super().retrieve(question, top_k=self.dense_candidates)
        dense_ids = [hit["chunk_id"] for hit in dense_hits]
        dense_rank = {chunk_id: pos for pos, chunk_id in enumerate(dense_ids, start=1)}

        bm25_scores = np.asarray(self._bm25.get_scores(tokenizar_bm25(question)))
        bm25_indices = [int(i) for i in np.argsort(-bm25_scores) if bm25_scores[i] > 0]
        bm25_ids = [self._chunks[i].id for i in bm25_indices[:self.bm25_candidates]]
        bm25_rank = {chunk_id: pos for pos, chunk_id in enumerate(bm25_ids, start=1)}

        rrf_scores = fusion_rrf([dense_ids, bm25_ids], k_rrf=self.rrf_k)
        ordenados = sorted(
            rrf_scores,
            key=lambda chunk_id: (-rrf_scores[chunk_id], dense_rank.get(chunk_id, 10**9),
                                  bm25_rank.get(chunk_id, 10**9)),
        )[:top_k]
        return [
            {"score": rrf_scores[chunk_id], "chunk_id": chunk_id,
             "document": self._chunk_by_id[chunk_id].document,
             "text": self._chunk_by_id[chunk_id].text,
             "dense_rank": dense_rank.get(chunk_id), "bm25_rank": bm25_rank.get(chunk_id)}
            for chunk_id in ordenados
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="Pregunta para consultar")
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--dense-candidates", type=int, default=20)
    parser.add_argument("--bm25-candidates", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    pipeline = HybridRagPipeline(args.dense_candidates, args.bm25_candidates, args.rrf_k)
    chunks = pipeline.ingest(args.corpus)
    if not chunks:
        raise SystemExit(f"ningún fragmento indexable en {args.corpus}/")
    pipeline.index(chunks)
    question = " ".join(args.question) or "REEMPLAZAR por una pregunta de prueba"
    for hit in pipeline.retrieve(question):
        print(f"  RRF={hit['score']:.4f} denso={hit['dense_rank']} BM25={hit['bm25_rank']}  "
              f"{hit['document']} / {hit['chunk_id']}  {hit['text'][:80]}…")
    salida = pipeline.answer(question)
    print(f"\n[{salida['generator']}] abstuvo={salida['abstained']}\n{salida['answer'][:1200]}")
