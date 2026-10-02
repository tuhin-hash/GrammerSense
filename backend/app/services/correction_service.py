"""
End-to-end correction orchestration service.
Coordinates candidate generation, ranking, alignment, classification, and diff generation.
"""

from typing import Dict, Any, List, Optional
import difflib

from backend.app.models.grammar_model import GrammarModel
from backend.app.correction.candidate_selector import CandidateSelector, CandidateCorrection
from backend.app.preprocessing.span_aligner import SpanAligner
from backend.app.services.detection_service import DetectionService, DetectedError


class CorrectionService:
    """Orchestrates candidate generation, selection, alignment, and classification."""

    def __init__(self, model: Optional[GrammarModel] = None):
        self.model = model or GrammarModel.get_instance()

    def process_text(
        self,
        text: str,
        num_candidates: int = 3,
        num_beams: int = 4,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Executes full GEC pipeline on input text.
        
        Returns:
            Structured dictionary with:
            - original_text
            - corrected_text
            - candidates: list of CandidateCorrection dicts
            - detected_errors: list of DetectedError dicts
            - diff_chunks: formatted visual diff segments for UI
            - metrics: error density, counts, and justified token ratios
        """
        cleaned_source = text.strip()
        if not cleaned_source:
            return {
                "original_text": text,
                "corrected_text": text,
                "candidates": [],
                "detected_errors": [],
                "diff_chunks": [{"type": "equal", "text": text}],
                "metrics": {
                    "original_word_count": 0,
                    "corrected_word_count": 0,
                    "error_count": 0,
                    "directly_identified_count": 0,
                    "inferred_count": 0,
                    "edit_density_per_100_words": 0.0,
                    "error_free_token_ratio": 1.0,
                    "model_confidence": 1.0,
                },
            }

        # 1. Candidate Generation
        raw_candidates = self.model.generate(
            input_text=cleaned_source,
            num_candidates=num_candidates,
            num_beams=num_beams,
            temperature=temperature,
        )

        # 2. Candidate Selection & Ranking
        best_cand, all_cands = CandidateSelector.select_best(
            source_text=cleaned_source,
            raw_candidates=raw_candidates,
        )
        corrected_text = best_cand.candidate_text

        # 3. Error Localization (Span Alignment)
        edit_spans = SpanAligner.align(
            source_text=cleaned_source,
            target_text=corrected_text
        )

        # 4. Error Classification (Directly Identified vs Inferred)
        detected_errors = DetectionService.detect_and_classify(
            source_text=cleaned_source,
            target_text=corrected_text,
            edit_spans=edit_spans
        )

        # 5. Visual Diff Generation
        diff_chunks = self._generate_visual_diff(cleaned_source, corrected_text)

        # 6. Technically Justified Quality Metrics
        src_words = cleaned_source.split()
        tgt_words = corrected_text.split()
        src_word_count = len(src_words)
        tgt_word_count = len(tgt_words)
        err_count = len(detected_errors)

        direct_count = sum(1 for e in detected_errors if e.verification_tier == "DIRECTLY_IDENTIFIED")
        inferred_count = sum(1 for e in detected_errors if e.verification_tier == "INFERRED")

        edit_density = round((err_count / max(1, src_word_count)) * 100, 2)
        # Error-free token ratio: proportion of words in source not requiring correction
        error_free_ratio = round(max(0.0, 1.0 - (err_count / max(1, src_word_count))), 4)

        return {
            "original_text": cleaned_source,
            "corrected_text": corrected_text,
            "candidates": [c.to_dict() for c in all_cands],
            "detected_errors": [e.to_dict() for e in detected_errors],
            "diff_chunks": diff_chunks,
            "metrics": {
                "original_word_count": src_word_count,
                "corrected_word_count": tgt_word_count,
                "error_count": err_count,
                "directly_identified_count": direct_count,
                "inferred_count": inferred_count,
                "edit_density_per_100_words": edit_density,
                "error_free_token_ratio": error_free_ratio,
                "model_confidence": best_cand.confidence,
            },
        }

    @staticmethod
    def _generate_visual_diff(source: str, target: str) -> List[Dict[str, str]]:
        """
        Creates fine-grained word diff chunks for frontend visual display.
        Chunks are labeled: 'equal', 'delete' (original removed), 'insert' (newly added).
        """
        s_words = source.split()
        t_words = target.split()
        matcher = difflib.SequenceMatcher(None, s_words, t_words)

        chunks: List[Dict[str, str]] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                chunks.append({"type": "equal", "text": " ".join(s_words[i1:i2])})
            elif tag == "replace":
                chunks.append({"type": "delete", "text": " ".join(s_words[i1:i2])})
                chunks.append({"type": "insert", "text": " ".join(t_words[j1:j2])})
            elif tag == "delete":
                chunks.append({"type": "delete", "text": " ".join(s_words[i1:i2])})
            elif tag == "insert":
                chunks.append({"type": "insert", "text": " ".join(t_words[j1:j2])})

        return chunks
