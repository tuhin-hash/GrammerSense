# Grammar Error Detection & Correction — Backend Service

The backend is built with **FastAPI** and provides high-performance, asynchronous REST endpoints for grammar error detection, correction, multi-candidate ranking, and benchmark evaluation.

## Architecture & Modules

The backend architecture strictly separates concerns into dedicated modules:

- **`app/models/grammar_model.py`**: Sequence-to-Sequence neural model manager (wrapping Hugging Face T5) with thread-optimized CPU inference, beam search, and true sequence confidence extraction.
- **`app/preprocessing/text_cleaner.py`**: Whitespace normalization, sentence splitting, and offset-preserving tokenization.
- **`app/preprocessing/span_aligner.py`**: Dynamic programming sequence alignment (token-level Levenshtein) for precise error localization.
- **`app/services/detection_service.py`**: Error classification engine distinguishing between:
  - **`DIRECTLY_IDENTIFIED`**: Morphological and syntactic rules (e.g. Subject-Verb Agreement, Verb Tense, Articles, Prepositions, Spelling).
  - **`INFERRED`**: Inferred by neural sequence alignment and contextual language modeling.
- **`app/correction/candidate_selector.py`**: Multi-candidate beam decoding, minimal-edit preservation scoring, and ranking.
- **`app/services/correction_service.py`**: End-to-end pipeline orchestrator producing visual diff chunks and interpretable metrics.
- **`app/evaluation/metrics.py`**: Google GLEU (Napoles et al.), corpus BLEU, Token Precision, Recall, and $F_{0.5}$ score.
- **`app/services/evaluation_service.py`**: Runs held-out benchmark evaluations with full reproducibility logging.

## REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System status, active model, execution device (CPU/CUDA) |
| `POST` | `/api/correct` | End-to-end detection, correction, diffs, candidates, and metrics |
| `POST` | `/api/detect` | Localized error spans and classification tiers only |
| `POST` | `/api/candidates` | Multiple beam search candidate corrections with confidences |
| `POST` | `/api/evaluate` | Run evaluation on held-out test split or custom pairs |
| `GET` | `/docs` | Interactive Swagger UI documentation |

## Running the Backend

```powershell
# From the backend directory or project root
cd grammar-error-detection-correction/backend
..\.venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
