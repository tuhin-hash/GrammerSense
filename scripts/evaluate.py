"""
Benchmark evaluation CLI runner for Grammar Error Correction.
Runs active model on held-out test set, calculates GLEU, BLEU, F0.5, and Exact Match,
and outputs structured JSON and Markdown reports.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.config import settings
from backend.app.models.grammar_model import GrammarModel
from backend.app.services.evaluation_service import EvaluationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def generate_markdown_report(report: dict, output_path: Path):
    """Writes a clean GitHub-flavored markdown summary of evaluation metrics."""
    m = report["metrics"]
    r = report["reproducibility"]

    md = f"""# GEC Benchmark Evaluation Report

**Generated**: `{report['timestamp']}`

## 1. Reproducibility Configuration
| Parameter | Value |
|---|---|
| **Model Checkpoint** | `{r['model_name_or_path']}` |
| **Execution Device** | `{r['device']}` |
| **Random Seed** | `{r['random_seed']}` |
| **Beam Search Size** | `{r['num_beams']}` |
| **Dataset Path** | `{r['dataset_path']}` |
| **Evaluated Test Samples** | `{r['num_evaluated_samples']}` |

## 2. Quantitative Benchmark Results
| Metric | Score | Description |
|---|---|---|
| **Corpus GLEU** | **{m['corpus_gleu']}%** | Google GLEU (JFLEG standard, rewards correct edits) |
| **Baseline GLEU** | {m['baseline_source_gleu']}% | Score if zero edits are performed |
| **GLEU Improvement** | **+{m['gleu_improvement']}%** | Net grammatical fluency gain |
| **Corpus BLEU** | {m['corpus_bleu']}% | 4-gram precision with brevity penalty |
| **Token Precision** | {m['token_precision_percent']}% | Accuracy of suggested edits |
| **Token Recall** | {m['token_recall_percent']}% | Proportion of gold errors corrected |
| **Token $F_{{0.5}}$ Score** | **{m['token_f0_5_score_percent']}%** | Standard GEC metric (Precision weighted 2x over Recall) |
| **Exact Match (EM)** | {m['exact_match_accuracy_percent']}% | Full-sentence exact match with reference |

## 3. Edit Classification Breakdown
- **True Positives (Correct Edits)**: `{m['edit_counts']['true_positives']}`
- **False Positives (Spurious Edits)**: `{m['edit_counts']['false_positives']}`
- **False Negatives (Missed Edits)**: `{m['edit_counts']['false_negatives']}`

## 4. Sample Test Predictions
| # | Source Sentence | Model Hypothesis | Gold Reference | GLEU |
|---|---|---|---|---|
"""
    for idx, s in enumerate(report.get("sample_results", [])[:10]):
        md += f"| {idx+1} | {s['source']} | **{s['hypothesis']}** | {s['target']} | {s['gleu']} |\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"Markdown report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GEC model on held-out test data.")
    parser.add_argument("--data_path", type=Path, default=settings.DATA_DIR / "processed" / "test.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", type=Path, default=settings.REPORTS_DIR / "latest_evaluation_report.json")
    parser.add_argument("--output_md", type=Path, default=settings.REPORTS_DIR / "benchmark_summary.md")

    args = parser.parse_args()

    # Fallback to demo dataset if processed test set doesn't exist
    if not args.data_path.exists():
        demo_path = settings.DATA_DIR / "demo" / "demo_pairs.jsonl"
        logger.warning(f"{args.data_path} not found. Falling back to {demo_path}")
        args.data_path = demo_path

    model = GrammarModel(model_name_or_path=args.model_name_or_path)
    eval_service = EvaluationService(model=model)

    logger.info(f"Running evaluation on {args.data_path}...")
    report = eval_service.evaluate_dataset(
        dataset_path=args.data_path,
        max_samples=args.max_samples,
        save_report=True,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    generate_markdown_report(report, args.output_md)

    print("\n" + "="*60)
    print("           ACTUAL GEC BENCHMARK EVALUATION RESULTS")
    print("="*60)
    m = report["metrics"]
    print(f"  Model Checkpoint        : {report['reproducibility']['model_name_or_path']}")
    print(f"  Test Samples Evaluated  : {report['reproducibility']['num_evaluated_samples']}")
    print(f"  Corpus GLEU Score       : {m['corpus_gleu']}%")
    print(f"  Corpus BLEU Score       : {m['corpus_bleu']}%")
    print(f"  Token Precision         : {m['token_precision_percent']}%")
    print(f"  Token Recall            : {m['token_recall_percent']}%")
    print(f"  Token F0.5 Score        : {m['token_f0_5_score_percent']}%")
    print(f"  Exact Match Accuracy    : {m['exact_match_accuracy_percent']}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
