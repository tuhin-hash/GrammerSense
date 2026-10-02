# Implementation Plan: End-to-End Grammar Error Detection & Correction System

Build a production-grade, real NLP system from scratch for **Grammar Error Detection (GED)** and **Grammar Error Correction (GEC)** inside `grammar-error-detection-correction/` with a modular backend, deep learning sequence-to-sequence model, rule-guided error span categorization, candidate selection, rigorous evaluation metrics, modern web frontend, automated tests, and documentation.

## User Review Required

> [!IMPORTANT]
> - **Python Version**: System has Python 3.12 (`py -3.12`), which is fully compatible with PyTorch and Hugging Face Transformers on Windows. We will use Python 3.12 to set up the dedicated virtual environment `.venv`.
> - **Hardware / Compute**: The machine runs in CPU mode (no CUDA GPU). We will configure the pipeline with a high-performance lightweight Seq2Seq model (such as `vennify/t5-base-grammar-correction` or fine-tuned `t5-small`) for fast, responsive CPU inference (<1-2s per sentence), while providing complete training scripts with PyTorch Seq2SeqTrainer for training/fine-tuning.
> - **Zero Placeholders**: All components will be real—real transformer neural generation, real alignment/span detection algorithms, real GLEU/BLEU/F0.5 evaluation, real FastAPI server, and a fully interactive React/Vite UI.

---

## Proposed Architecture

```
grammar-error-detection-correction/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI application entrypoint & middleware
│   │   ├── config.py                   # Pydantic Settings, model paths, hyperparams
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py               # REST API endpoints (/correct, /detect, /candidates, /evaluate, /health)
│   │   │   └── schemas.py              # Pydantic request & response schemas
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── correction_service.py   # Seq2Seq inference & multi-candidate generation
│   │   │   ├── detection_service.py    # Span-level alignment, classification & linguistic rules
│   │   │   └── evaluation_service.py   # GLEU, BLEU, F0.5 & edit-distance metrics
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── grammar_model.py        # Hugging Face T5/Seq2Seq wrapper with CPU optimizations
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── text_cleaner.py         # Normalization, tokenization, sentence splitting
│   │   │   └── span_aligner.py         # Dynamic programming / Levenshtein word & char alignment
│   │   ├── correction/
│   │   │   ├── __init__.py
│   │   │   └── candidate_selector.py   # Multi-candidate beam search & confidence ranking
│   │   └── evaluation/
│   │       ├── __init__.py
│   │       ├── metrics.py              # GLEU, BLEU, Token Precision, Recall, F0.5
│   │       └── report_generator.py     # Evaluation summary and benchmark generator
│   ├── requirements.txt
│   └── README.md
│
├── frontend/                           # React + Vite + TailwindCSS Modern UI
│   ├── src/
│   │   ├── App.jsx                     # Main application layout with tabs & stats
│   │   ├── components/
│   │   │   ├── Header.jsx              # App bar, model status indicator, theme
│   │   │   ├── Editor.jsx              # Live interactive input with error badges & inline highlights
│   │   │   ├── DiffViewer.jsx          # Word/char level visual diff (Green/Red insertions/deletions)
│   │   │   ├── ErrorList.jsx           # Categorized error cards with "Apply" and explanation
│   │   │   ├── CandidatePicker.jsx     # Multiple candidate selections with confidence scores
│   │   │   ├── MetricsDisplay.jsx      # Readability, grammar quality gauge, counts
│   │   │   └── BenchmarkModal.jsx      # Run dataset evaluation & view GLEU/BLEU/F0.5 scores
│   │   ├── services/
│   │   │   └── api.js                  # Axios client for FastAPI endpoints
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── README.md
│
├── data/
│   ├── raw/                            # Downloaded benchmark datasets (e.g. JFLEG / synthetic GEC pairs)
│   ├── processed/                      # Preprocessed train/val/test jsonl splits
│   └── README.md
│
├── models/
│   ├── checkpoints/                    # Saved weights / downloaded model cache
│   └── README.md
│
├── scripts/
│   ├── download_data.py                # Download public JFLEG or benchmark datasets
│   ├── prepare_data.py                 # Clean, pair, validate and create train/val/test splits
│   ├── train.py                        # Fine-tuning Seq2Seq T5 on GEC pairs with PyTorch
│   ├── evaluate.py                     # CLI benchmark evaluation generating GLEU & F0.5 reports
│   └── inference.py                    # Interactive CLI & batch inference tool
│
├── tests/
│   ├── test_preprocessing.py           # Text cleaning and span alignment tests
│   ├── test_detection.py               # Token/span error detection and category classification
│   ├── test_correction.py              # Model inference and candidate selection
│   ├── test_evaluation.py              # GLEU, BLEU, Precision, Recall, F0.5 score tests
│   └── test_api.py                     # FastAPI test client for all REST endpoints
│
├── reports/
│   └── baseline_evaluation.json        # Evaluation metrics output
├── notebooks/
│   └── gec_exploration.ipynb           # Interactive exploration notebook
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Detailed Step-by-Step Implementation

### Phase 1: Environment & Project Foundation (Steps 1–5)
1. **Initialize Directory**: Create `C:\Users\tuhin\Documents\Projects\grammar-error-detection-correction` and full directory hierarchy.
2. **Setup Virtual Environment**: Create a virtual environment with `py -3.12 -m venv .venv`.
3. **Dependency Specification**:
   - `torch` (CPU wheel), `transformers`, `sentencepiece`, `accelerate`
   - `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`
   - `nltk`, `scikit-learn`, `numpy`
   - `pytest`, `httpx`, `requests`
   - Pin reliable versions to avoid platform build compilation errors on Windows.
4. **Install & Verify**: Activate `.venv` and install backend dependencies.

### Phase 2: Core NLP Engine (Steps 6–10)
1. **Seq2Seq Grammar Correction Model (`models/grammar_model.py`)**:
   - Wrap Hugging Face `AutoModelForSeq2SeqLM` and `AutoTokenizer` (loading `vennify/t5-base-grammar-correction` or fine-tuned `t5-small`).
   - Implement beam search decoding, temperature control, top-k/top-p sampling for multi-candidate generation.
   - Built-in CPU inference optimizations (torch inference mode, caching, input length trimming).
2. **Span-Level Error Detection & Classification (`preprocessing/span_aligner.py`, `services/detection_service.py`)**:
   - Implement sequence alignment (Levenshtein diff / dynamic programming on token level).
   - Detect insertion, deletion, and replacement operations with exact character offset tracking (`start_char`, `end_char`).
   - Categorize errors into linguistic classes:
     - `Subject-Verb Agreement` (e.g., "he go" -> "he goes")
     - `Verb Tense / Form` (e.g., "have went" -> "have gone")
     - `Article / Determiner` (e.g., "an book" -> "a book")
     - `Preposition` (e.g., "interested on" -> "interested in")
     - `Spelling / Typo` (e.g., "teh" -> "the", edit distance <= 2)
     - `Punctuation & Capitalization` (e.g., missing period, lowercase sentence starts)
     - `Word Order / Redundancy`
   - Generate contextual pedagogical explanations for each error.
3. **Candidate Selection & Ranking (`correction/candidate_selector.py`)**:
   - Generate top-$N$ candidate sentences using diverse beam search.
   - Rank candidates based on model generation log-probabilities, edit-distance penalty, and fluency heuristics.

### Phase 3: Dataset, Training & Evaluation Pipelines (Steps 11–14)
1. **Dataset Pipeline (`scripts/download_data.py`, `scripts/prepare_data.py`)**:
   - Download or generate standard GEC benchmark datasets (including the canonical JFLEG test/dev set and rich synthetic GEC error-correction pairs).
   - Format into JSONL schema: `{"id": str, "source": str, "target": str, "edits": [...]}`.
   - Partition into train (80%), validation (10%), test (10%).
2. **Model Training / Fine-Tuning (`scripts/train.py`)**:
   - Full Seq2Seq training script supporting arguments `--model_name`, `--dataset_dir`, `--epochs`, `--batch_size`, `--learning_rate`, `--max_source_length`, `--max_target_length`, `--output_dir`.
   - Supports dry-run / sample training mode for immediate CPU verification.
3. **Evaluation Framework (`backend/app/evaluation/metrics.py`, `scripts/evaluate.py`)**:
   - Exact implementation of GEC standard metrics:
     - **GLEU (Generalized Language Evaluation Understanding)**: Sentence and corpus-level $n$-gram precision/recall against reference corrections.
     - **BLEU Score**: Multi-reference corpus BLEU.
     - **Word-level Precision, Recall, and $F_{0.5}$ score**: Weights precision higher than recall as standard in CoNLL/BEA shared tasks.
     - **Edit Distance Reduction Rate**: Measures how much error entropy was reduced.
   - Output formatted JSON and Markdown benchmark reports.

### Phase 4: Backend REST API (Steps 15–16)
1. **FastAPI Application (`backend/app/main.py`, `backend/app/api/routes.py`)**:
   - Endpoint `/api/health`: Health status, model name, device, memory info.
   - Endpoint `/api/correct`: Primary endpoint returning corrected text, span-level detected errors with explanations, visual diff chunks, and grammar score.
   - Endpoint `/api/detect`: Span-level detection only without full rewrite.
   - Endpoint `/api/candidates`: Returns multiple candidate corrections with confidence scores.
   - Endpoint `/api/evaluate`: Runs evaluation on a submitted test set or built-in validation pairs.
   - CORS middleware configured for frontend communication.

### Phase 5: Modern Interactive Frontend (Step 17)
1. **React + Vite Setup (`frontend/`)**:
   - Setup lightweight Vite + React app with modern UI styling.
   - Key UI features:
     - **Interactive Live Editor**: Highlights errors directly in the text with color-coded underlines; click any error to view category, explanation, and "Apply Correction".
     - **Visual Diff Comparison**: Side-by-side or inline view showing deletions (strikethrough red) and additions (highlighted green).
     - **Candidate Carousel**: Allows users to inspect and toggle between top AI candidates.
     - **Grammar & Readability Score Card**: Real-time score gauge (0–100), word count, detected error tally.
     - **Preset Examples**: 1-click test sentences covering common grammar errors.
     - **Evaluation Dashboard Modal**: View live benchmark metrics (GLEU, BLEU, $F_{0.5}$) and test custom pairs.

### Phase 6: Automated Testing & Verification (Steps 18–19)
1. **Pytest Suite (`tests/`)**:
   - `test_preprocessing.py`: Verify text normalization and character offset preservation.
   - `test_detection.py`: Verify detection of common grammatical errors (Subject-Verb, Spelling, Articles, Tense).
   - `test_correction.py`: Verify model generates valid corrections on ungrammatical inputs.
   - `test_evaluation.py`: Unit test GLEU, BLEU, and $F_{0.5}$ metric calculations against known reference vectors.
   - `test_api.py`: Test FastAPI endpoints using `TestClient`.
2. **System Verification**:
   - Run backend server and test with curl/HTTP requests.
   - Build frontend assets and verify production build.
   - Run the CLI inference script (`scripts/inference.py`).
   - Run evaluation script (`scripts/evaluate.py`).

### Phase 7: Documentation & Fresh Install Guide (Step 20)
1. Write top-level `README.md`, `backend/README.md`, `frontend/README.md`, `data/README.md`.
2. Include exact fresh-install commands for students/graders.
3. Document model architecture, error taxonomy, evaluation benchmark results, and troubleshooting tips.

---

## Verification Plan

### Automated Tests
```powershell
# Run backend pytest suite
.venv\Scripts\python -m pytest tests/ -v

# Run evaluation script on benchmark dataset
.venv\Scripts\python scripts/evaluate.py --data_path data/processed/test.jsonl

# Run CLI inference test
.venv\Scripts\python scripts/inference.py --text "She don't likes apples and have went to store."
```

### Manual & End-to-End Verification
1. **Backend Verification**: Send HTTP POST requests to `http://127.0.0.1:8000/api/correct` with various error types and assert JSON payload schema.
2. **Frontend Verification**: Start frontend dev server, navigate to UI, input sample erroneous sentences, verify inline error card rendering, diff view, candidate selection, and score gauge.
3. **Fresh Install Verification**: Ensure all steps are documented and cleanly runnable.
