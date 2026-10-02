"""
Evaluation orchestration service.
Runs held-out evaluation datasets through the active GEC model, computes benchmark metrics,
and logs complete reproducibility metadata.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.app.config import settings
from backend.app.models.grammar_model import GrammarModel
from backend.app.evaluation.metrics import GECMetrics


class EvaluationService:
    """Executes evaluation runs and formats benchmark reports."""

    def __init__(self, model: Optional[GrammarModel] = None):
        self.model = model or GrammarModel.get_instance()

    def evaluate_dataset(
        self,
        dataset_path: Path,
        max_samples: Optional[int] = None,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluates the active model on a JSONL dataset containing 'source' and 'target' pairs.
        Returns complete benchmark metrics with reproducibility metadata.
        """
        if not dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_path}")

        sources: List[str] = []
        references: List[str] = []
        sample_ids: List[str] = []

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                sample_ids.append(item.get("id", f"sample-{len(sources)}"))
                sources.append(item["source"])
                references.append(item["target"])
                if max_samples and len(sources) >= max_samples:
                    break

        if not sources:
            raise ValueError(f"Dataset at {dataset_path} contains no valid samples.")

        # Ensure model is ready
        if not self.model.is_loaded:
            self.model.load()

        # Generate model hypotheses
        hypotheses: List[str] = []
        sample_details: List[Dict[str, Any]] = []

        for sid, src, ref in zip(sample_ids, sources, references):
            raw_cands = self.model.generate(
                input_text=src,
                num_candidates=1,
                num_beams=settings.NUM_BEAMS,
            )
            hyp = raw_cands[0][0] if raw_cands else src
            hypotheses.append(hyp)

            sample_details.append({
                "id": sid,
                "source": src,
                "target": ref,
                "hypothesis": hyp,
                "gleu": GECMetrics.sentence_gleu(src, hyp, ref),
                "is_exact_match": (hyp.strip().lower() == ref.strip().lower()),
            })

        # Calculate benchmark metrics
        corpus_gleu = GECMetrics.corpus_gleu(sources, hypotheses, references)
        corpus_bleu = GECMetrics.corpus_bleu(hypotheses, references)
        f_beta_results = GECMetrics.compute_f_beta(
            sources, hypotheses, references, beta=settings.FBETA_BETA
        )
        exact_match = GECMetrics.compute_exact_match(hypotheses, references)

        # Baseline: score when making no edits (source == hypothesis)
        baseline_gleu = GECMetrics.corpus_gleu(sources, sources, references)

        report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reproducibility": {
                "model_name_or_path": self.model.active_checkpoint,
                "device": self.model.device,
                "random_seed": settings.RANDOM_SEED,
                "num_beams": settings.NUM_BEAMS,
                "max_input_length": settings.MAX_INPUT_LENGTH,
                "dataset_path": str(dataset_path),
                "num_evaluated_samples": len(sources),
            },
            "metrics": {
                "corpus_gleu": corpus_gleu,
                "baseline_source_gleu": baseline_gleu,
                "gleu_improvement": round(corpus_gleu - baseline_gleu, 2),
                "corpus_bleu": corpus_bleu,
                "exact_match_accuracy_percent": exact_match,
                "token_precision_percent": f_beta_results["precision"],
                "token_recall_percent": f_beta_results["recall"],
                "token_f0_5_score_percent": f_beta_results["f_beta"],
                "edit_counts": {
                    "true_positives": f_beta_results["true_positives"],
                    "false_positives": f_beta_results["false_positives"],
                    "false_negatives": f_beta_results["false_negatives"],
                },
            },
            "sample_results": sample_details,
        }

        if save_report:
            settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report_path = settings.REPORTS_DIR / "latest_evaluation_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return report

    def evaluate_pairs(self, pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluates ad-hoc pairs submitted via API or interactive tests."""
        sources = [p["source"] for p in pairs]
        references = [p["target"] for p in pairs]
        hypotheses = []

        if not self.model.is_loaded:
            self.model.load()

        for src in sources:
            cands = self.model.generate(src, num_candidates=1, num_beams=settings.NUM_BEAMS)
            hyp = cands[0][0] if cands else src
            hypotheses.append(hyp)

        gleu = GECMetrics.corpus_gleu(sources, hypotheses, references)
        bleu = GECMetrics.corpus_bleu(hypotheses, references)
        f_res = GECMetrics.compute_f_beta(sources, hypotheses, references, beta=settings.FBETA_BETA)
        em = GECMetrics.compute_exact_match(hypotheses, references)

        return {
            "num_samples": len(sources),
            "corpus_gleu": gleu,
            "corpus_bleu": bleu,
            "exact_match_accuracy": em,
            "f0_5_score": f_res["f_beta"],
            "precision": f_res["precision"],
            "recall": f_res["recall"],
            "pairs": [
                {
                    "source": s,
                    "target": r,
                    "hypothesis": h,
                    "is_exact_match": (h.strip().lower() == r.strip().lower()),
                }
                for s, r, h in zip(sources, references, hypotheses)
            ]
        }
