# Lab 02 — RAG con Vector Search y Evaluación

Andamiaje del Taller 2. Trae la ingesta, la fragmentación, el índice, la recuperación, el
prompt, **la llamada al LLM** y el evaluador; lo que el estudiante pone es el corpus, el
golden set, los parámetros con su justificación y el análisis.

> **Los embeddings son de `bge-m3`, servido en la H200 de la USFQ.** Hace falta la VPN
> GlobalProtect. Sin VPN, la alternativa es la API de OpenAI (`EMBEDDING_BACKEND=openai`).
> Cuatro cosas que no fallarían por sí solas y aquí fallan o se miden: el PDF escaneado que
> se indexaría vacío, el fragmento que se truncaría en silencio, las preguntas negativas que
> pondrían un techo al Hit Rate, y la abstención que ningún detector por igualdad de cadena
> podría ver. El detalle está en la cabecera de cada archivo.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env            # y pon la clave de OpenAI que reparte el curso, si la usas
# GlobalProtect conectada: los embeddings se calculan en la H200 (bge-m3)
docker run -p 6333:6333 qdrant/qdrant       # o QDRANT_URL=":memory:" para correr sin Docker
```

`QDRANT_URL=":memory:"` es la misma API sin contenedor: sirve para la Parte 0 y para
depurar. El sábado se evalúa contra el contenedor, que es lo que persiste el índice.

## Estructura

```
Lab-02-RAG-VectorSearch/
├── README.md
├── requirements.txt
├── .env.example
├── ingestion.py               Carga, parseo y fragmentación EN TOKENS del modelo
├── rag_pipeline.py            Ingesta → índice → recuperación → prompt → generación
├── evaluation.py              Hit Rate y MRR sobre respondibles; dos tasas de abstención; CSV crudo
├── golden_set_plantilla.json  Plantilla (10 preguntas, anotadas por documento y fragmento)
├── ejemplos/                  Corpus mínimo + un PDF escaneado, para la Parte 0 del taller
└── corpus/                    Tu corpus (lo creas tú)
```

## Baseline

1. Copia tus PDFs o `.txt`/`.md` en `corpus/`.
2. Completa `golden_set_plantilla.json` y guárdalo como `golden_set.json`.
3. `python rag_pipeline.py "una pregunta de prueba"`: ingesta, indexa, recupera y genera.
4. `python evaluation.py --k 5`: métricas y `resultados.csv`, una fila por consulta.

RAGAS, búsqueda híbrida y reranking son extensiones. Primero debe funcionar el baseline.

## Extensión B: búsqueda híbrida (BM25 + densa + RRF)

`hybrid_pipeline.py` conserva el corpus, embeddings, Qdrant y prompt del baseline. Añade
BM25 sobre los mismos chunks y fusiona el top-20 denso con el top-20 léxico mediante
Reciprocal Rank Fusion (RRF, `k=60`) antes de devolver el top-5. Sus puntajes son de RRF y
no se comparan directamente con los puntajes de coseno del baseline.

```bash
python hybrid_pipeline.py "¿Qué es un dispositivo lógico programable?"
python evaluation_hybrid.py --k 3 --csv resultados_hibrido_k3.csv
python evaluation_hybrid.py --k 5 --csv resultados_hibrido_k5.csv
```

Cada ejecución de `evaluation_hybrid.py` vuelve a indexar el corpus en Qdrant y construye
BM25 en memoria, para que ambos rankings correspondan exactamente a los mismos chunks.

## Las tres fallas que no fallan (Parte 0 del taller)

La Parte 0 corre con el MiniLM de los notebooks **en tu máquina** (`EMBEDDING_BACKEND=local`):
sin clave, sin VPN, sin Docker y en un minuto. Es la **única** parte del taller que no usa
`bge-m3`, y es a propósito: la 0.b necesita un modelo que trunque a 128 tokens.

```bash
EMBEDDING_BACKEND=local QDRANT_URL=":memory:" python rag_pipeline.py "¿cuántos días de vacaciones puedo transferir?"
```

- **El PDF escaneado.** `ejemplos/instructivo_escaneado.pdf` no tiene capa de texto. Sin
  comprobación se indexaría como un fragmento hecho de `[page=1] [page=2]`, sin ninguna
  excepción. La ingesta avisa: `0 caracteres útiles en 2 página(s) … NO se indexa`. Si tu
  corpus real trae uno, ese documento no está aunque el índice diga que sí.
- **El fragmento que se corta.** El MiniLM trunca a **128 tokens** de secuencia
  (`fuentes/modelos/modelos-2026-1.json`, fila `embed_notebook_s2`, verificado 2026-08-27),
  y lo hace sin avisar. La ingesta fragmenta en tokens del **mismo** tokenizador y lee el
  tope del modelo; si pides un fragmento mayor, avisa cuánto se pierde. Pide
  `chunk_tokens=900` y míralo.
- **El índice que no se queja.** Pregunta algo que no está (`"¿cuál es la política de
  mascotas?"`): devuelve cinco vecinos con puntajes de aspecto sano. Ningún Hit Rate
  detecta esto; por eso el golden set lleva negativas y el evaluador mide si el sistema
  **se abstiene**.

## El modelo de embeddings: `bge-m3` en la H200

`rag_pipeline.py` calcula los embeddings con **`bge-m3`** (fila `embed_local_multilingue` de
la tabla semestral): multilingüe —el corpus por defecto del taller está en español—, 1024
dimensiones y un tope de **8192 tokens**. Lo sirve el Ollama de la H200 de la USFQ
(`H200_EMBED_URL`, por defecto `http://172.28.230.10:11434`); el cliente busca en su catálogo
el id servido que empieza por `bge-m3`, en vez de escribirlo, y le pide `truncate: false`:
si un texto no cabe, el servidor da error en vez de recortarlo en silencio. El tokenizador
con el que se fragmenta es el de `BAAI/bge-m3` —se descarga solo el tokenizador—, para que
la ingesta cuente en los mismos tokens que el servidor.

El fragmento por defecto es de **512 tokens** con un quinto de solapamiento: con 8192 de
tope, «el fragmento más grande que cabe» ya no es un buen valor por defecto, y 512 es el
tamaño de las notas del curso. Cámbialo con `chunk_tokens` y justifícalo en el informe.

| `EMBEDDING_BACKEND` | Modelo (fila) | Tope | Hace falta |
|---|---|---|---|
| `h200` (por defecto) | `bge-m3` (`embed_local_multilingue`) | 8192 | VPN GlobalProtect |
| `openai` | `text-embedding-3-small` (`embed_api_economico`); `EMBEDDING_MODEL` elige otro | 8192 | `OPENAI_API_KEY` |
| `local` | MiniLM multilingüe (`embed_notebook_s2`) | 128 | nada; **solo para la Parte 0** |

**Sin VPN, la alternativa es OpenAI**, y cuesta poco: los embeddings de un corpus de 50
páginas son unas decenas de miles de tokens. Lo que no se puede es **mezclar**: un índice
construido con un modelo solo se consulta con ese mismo modelo, así que si cambias de ruta,
reindexa. Y se declara en el informe con qué modelo se construyó el índice.

## La generación, y la abstención

`rag_pipeline.generate` elige la ruta por la clave disponible: `OPENAI_API_KEY` → fila
`propietario_economico`; `ANTHROPIC_API_KEY` → fila `juez_economico`; ninguna → Ollama con
la fila `open_weight_pequeno`; y sin Ollama, «modo inspección», que devuelve el prompt.
`GENERATION_MODEL` sobreescribe el modelo. Ningún nombre está escrito a mano en el código:
salen de la tabla semestral por su `id`.

La frase de abstención es **una** en todo el curso: `ABSTENCION` en `rag_pipeline.py`, la
misma en el prompt, en el golden set y en el notebook del miércoles. `se_abstuvo()` la
detecta **normalizada** (minúsculas, sin tildes, sin puntuación): con una tilde de más o un
punto final de menos, un detector por igualdad de cadena no dispararía nunca.

## Qué mide `evaluation.py`

| Métrica | Sobre qué | Qué significa |
|---|---|---|
| `hit_rate` | preguntas **respondibles** | ¿algún fragmento recuperado pertenece al documento fuente (y contiene el fragmento esperado, si lo diste)? |
| `mrr` | respondibles | media de 1/posición del primer acierto |
| `abstencion_correcta` | negativas | fracción en la que el sistema se abstuvo — lo correcto |
| `abstencion_indebida` | respondibles | fracción en la que se abstuvo — lo incorrecto |

Las negativas **no entran** en `hit_rate` ni en `mrr`: no hay nada que recuperar. Con la
plantilla anterior sí entraban, y un sistema perfecto reportaba 0,70 como techo.
`--sin-generar` salta el LLM: las dos tasas salen «sin medir» y las otras dos, igual.

El golden set se anota **por documento y por fragmento literal**, no por `chunk_id`: los
identificadores cambian al re-fragmentar y la Opción A del taller los invalidaba sin fallar.

## Recursos

- Qdrant docs: https://qdrant.tech/documentation (consultado 2026-09-19: `create_collection`
  + `query_points`; HNSW es el único índice denso, `m=16`, `ef_construct=100`)
- sentence-transformers: https://www.sbert.net
- RAGAS: https://docs.ragas.io
