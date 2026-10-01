"""Evalúa la extensión híbrida BM25 + densa + RRF con el golden set del baseline.

Ejemplos:
    python evaluation_hybrid.py --k 3 --csv resultados_hibrido_k3.csv
    python evaluation_hybrid.py --k 5 --csv resultados_hibrido_k5.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation import _fmt, escribir_csv, evaluate_retrieval, load_golden_set
from hybrid_pipeline import HybridRagPipeline
from rag_pipeline import EMBEDDING_MODEL


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=Path("golden_set.json"))
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--csv", type=Path, default=Path("resultados_hibrido.csv"))
    parser.add_argument("--sin-generar", action="store_true",
                        help="solo recuperación: no llama al LLM y no mide abstención")
    parser.add_argument("--dense-candidates", type=int, default=20)
    parser.add_argument("--bm25-candidates", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    pipeline = HybridRagPipeline(args.dense_candidates, args.bm25_candidates, args.rrf_k)
    chunks = pipeline.ingest(args.corpus)
    if not chunks:
        raise SystemExit(f"ningún fragmento indexable en {args.corpus}/")
    pipeline.index(chunks)
    report = evaluate_retrieval(load_golden_set(args.golden), pipeline, k=args.k,
                                generar=not args.sin_generar)
    escribir_csv(report["rows"], args.csv, EMBEDDING_MODEL)

    print(f"híbrido: denso top-{args.dense_candidates} + BM25 top-{args.bm25_candidates} "
          f"+ RRF k={args.rrf_k}")
    print(f"golden set: {report['n_preguntas']} preguntas ({report['n_respondibles']} respondibles, "
          f"{report['n_negativas']} negativas) · k = {report['k']} · modelo {EMBEDDING_MODEL}")
    print(f"Hit Rate@{args.k} (respondibles): {_fmt(report['hit_rate'])}")
    print(f"MRR (respondibles):          {_fmt(report['mrr'])}")
    print(f"abstención correcta:         {_fmt(report['abstencion_correcta'])}   (negativas)")
    print(f"abstención indebida:         {_fmt(report['abstencion_indebida'])}   (respondibles)")
    for tipo, metricas in report["por_tipo"].items():
        print(f"  {tipo:<12} n={metricas['n']}  hit={_fmt(metricas['hit_rate'])}  "
              f"mrr={_fmt(metricas['mrr'])}  abstuvo={_fmt(metricas['abstuvo'])}")
    print(f"filas crudas en {args.csv}")
