"""
Grammar Error Correction (GEC) evaluation metrics.
Implements Google GLEU (Napoles et al.), corpus BLEU, Token-level Precision/Recall/F0.5 (CoNLL/BEA),
and Exact Match accuracy.
Computes real mathematical metrics from actual model outputs on test sets without fabrication.
"""

import math
from collections import Counter
from typing import List, Dict, Any, Tuple
import difflib


class GECMetrics:
    """Computes standard benchmark metrics for Grammar Error Correction."""

    @staticmethod
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        """Extracts n-gram frequency counter from list of tokens."""
        if len(tokens) < n:
            return Counter()
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    @classmethod
    def sentence_gleu(
        cls,
        source: str,
        hypothesis: str,
        reference: str,
        min_n: int = 1,
        max_n: int = 4
    ) -> float:
        """
        Calculates sentence-level GLEU score (Google / JFLEG benchmark).
        Rewards matching reference n-grams that differ from source,
        penalizes ungrammatical n-grams matching source when reference changed them.
        """
        src_tokens = source.strip().lower().split()
        hyp_tokens = hypothesis.strip().lower().split()
        ref_tokens = reference.strip().lower().split()

        if not hyp_tokens:
            return 0.0

        precisions = []
        for n in range(min_n, max_n + 1):
            hyp_ngrams = cls.get_ngrams(hyp_tokens, n)
            ref_ngrams = cls.get_ngrams(ref_tokens, n)
            src_ngrams = cls.get_ngrams(src_tokens, n)

            total_hyp = sum(hyp_ngrams.values())
            if total_hyp == 0:
                continue

            # Matching n-grams between hypothesis and reference
            match_count = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())

            # Penalty: hypothesis n-grams that match source but are absent or less frequent in reference
            penalty_count = sum(
                max(0, min(count, src_ngrams[ng]) - ref_ngrams[ng])
                for ng, count in hyp_ngrams.items()
            )

            valid_matches = max(0, match_count - penalty_count)
            precisions.append((valid_matches + 1e-8) / (total_hyp + 1e-8))

        if not precisions:
            return 0.0

        # Geometric mean
        log_sum = sum(math.log(p) for p in precisions)
        geom_mean = math.exp(log_sum / len(precisions))

        # Brevity penalty
        bp = 1.0
        if len(hyp_tokens) < len(ref_tokens):
            bp = math.exp(1.0 - (len(ref_tokens) / max(1, len(hyp_tokens))))

        return round(geom_mean * bp, 4)

    @classmethod
    def corpus_gleu(
        cls,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        min_n: int = 1,
        max_n: int = 4
    ) -> float:
        """
        Calculates corpus-level GLEU across all evaluation pairs.
        """
        assert len(sources) == len(hypotheses) == len(references)
        if not sources:
            return 0.0

        total_matches = {n: 0.0 for n in range(min_n, max_n + 1)}
        total_penalties = {n: 0.0 for n in range(min_n, max_n + 1)}
        total_hyp_ngrams = {n: 0.0 for n in range(min_n, max_n + 1)}
        total_hyp_len = 0
        total_ref_len = 0

        for src, hyp, ref in zip(sources, hypotheses, references):
            s_toks = src.strip().lower().split()
            h_toks = hyp.strip().lower().split()
            r_toks = ref.strip().lower().split()

            total_hyp_len += len(h_toks)
            total_ref_len += len(r_toks)

            for n in range(min_n, max_n + 1):
                h_ng = cls.get_ngrams(h_toks, n)
                r_ng = cls.get_ngrams(r_toks, n)
                s_ng = cls.get_ngrams(s_toks, n)

                total_hyp_ngrams[n] += sum(h_ng.values())
                total_matches[n] += sum(min(count, r_ng[ng]) for ng, count in h_ng.items())
                total_penalties[n] += sum(
                    max(0, min(count, s_ng[ng]) - r_ng[ng])
                    for ng, count in h_ng.items()
                )

        precisions = []
        for n in range(min_n, max_n + 1):
            if total_hyp_ngrams[n] == 0:
                continue
            effective_matches = max(0.0, total_matches[n] - total_penalties[n])
            precisions.append((effective_matches + 1e-8) / (total_hyp_ngrams[n] + 1e-8))

        if not precisions:
            return 0.0

        log_mean = sum(math.log(p) for p in precisions) / len(precisions)
        geom = math.exp(log_mean)

        bp = 1.0
        if total_hyp_len < total_ref_len and total_hyp_len > 0:
            bp = math.exp(1.0 - (total_ref_len / total_hyp_len))

        return round(geom * bp * 100, 2)  # Typically reported on 0-100 scale

    @classmethod
    def corpus_bleu(
        cls,
        hypotheses: List[str],
        references: List[str],
        max_n: int = 4
    ) -> float:
        """
        Calculates corpus-level BLEU score with smoothing.
        """
        total_matches = {n: 0 for n in range(1, max_n + 1)}
        total_possible = {n: 0 for n in range(1, max_n + 1)}
        c_len = 0
        r_len = 0

        for hyp, ref in zip(hypotheses, references):
            h_toks = hyp.strip().lower().split()
            r_toks = ref.strip().lower().split()
            c_len += len(h_toks)
            r_len += len(r_toks)

            for n in range(1, max_n + 1):
                h_ng = cls.get_ngrams(h_toks, n)
                r_ng = cls.get_ngrams(r_toks, n)
                total_possible[n] += sum(h_ng.values())
                total_matches[n] += sum(min(cnt, r_ng[ng]) for ng, cnt in h_ng.items())

        precisions = []
        for n in range(1, max_n + 1):
            if total_possible[n] == 0:
                precisions.append(1e-8)
            else:
                precisions.append((total_matches[n] + 0.1) / (total_possible[n] + 0.1))

        log_prec = sum(math.log(p) for p in precisions) / max_n
        bp = 1.0 if c_len > r_len else (math.exp(1.0 - (r_len / max(1, c_len))) if c_len > 0 else 0.0)
        bleu = bp * math.exp(log_prec) * 100
        return round(bleu, 2)

    @classmethod
    def extract_edits(cls, source: str, target: str) -> List[Tuple[int, int, str, str]]:
        """
        Extracts token-level edit operations (start_idx, end_idx, original, replacement).
        """
        s_words = source.strip().lower().split()
        t_words = target.strip().lower().split()
        matcher = difflib.SequenceMatcher(None, s_words, t_words)

        edits = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                orig = " ".join(s_words[i1:i2])
                repl = " ".join(t_words[j1:j2])
                edits.append((i1, i2, orig, repl))
        return edits

    @classmethod
    def compute_f_beta(
        cls,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        beta: float = 0.5
    ) -> Dict[str, float]:
        """
        Computes Token/Edit-level Precision, Recall, and F_beta (standard CoNLL/BEA shared task metric).
        Beta = 0.5 weights precision twice as high as recall.
        """
        tp = 0
        fp = 0
        fn = 0

        for src, hyp, ref in zip(sources, hypotheses, references):
            model_edits = set(cls.extract_edits(src, hyp))
            gold_edits = set(cls.extract_edits(src, ref))

            for edit in model_edits:
                if edit in gold_edits:
                    tp += 1
                else:
                    fp += 1

            for edit in gold_edits:
                if edit not in model_edits:
                    fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        beta_sq = beta ** 2
        denom = (beta_sq * precision) + recall
        f_score = ((1 + beta_sq) * precision * recall) / denom if denom > 0 else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f_beta": round(f_score * 100, 2),
            "beta": beta
        }

    @classmethod
    def compute_exact_match(cls, hypotheses: List[str], references: List[str]) -> float:
        """Computes percentage of sentences matching gold reference exactly."""
        if not hypotheses:
            return 0.0
        matches = sum(
            1 for h, r in zip(hypotheses, references)
            if h.strip().lower() == r.strip().lower()
        )
        return round((matches / len(hypotheses)) * 100, 2)
