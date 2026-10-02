"""
Unit tests for GEC benchmark metrics: Google GLEU, BLEU, F0.5, and Exact Match.
"""

import pytest
from backend.app.evaluation.metrics import GECMetrics


def test_sentence_gleu_identical():
    src = "She goes to college every day."
    hyp = "She goes to college every day."
    ref = "She goes to college every day."
    score = GECMetrics.sentence_gleu(src, hyp, ref)
    # Identical sentence should yield 1.0 (or 100%)
    assert score == 1.0


def test_sentence_gleu_improvement():
    src = "She go to college every day."
    good_hyp = "She goes to college every day."
    bad_hyp = "She going to college every day."
    ref = "She goes to college every day."

    good_score = GECMetrics.sentence_gleu(src, good_hyp, ref)
    bad_score = GECMetrics.sentence_gleu(src, bad_hyp, ref)

    assert good_score > bad_score


def test_corpus_gleu():
    sources = ["She go to college.", "He have two brothers."]
    hypotheses = ["She goes to college.", "He has two brothers."]
    references = ["She goes to college.", "He has two brothers."]

    corpus_score = GECMetrics.corpus_gleu(sources, hypotheses, references)
    assert corpus_score == 100.0


def test_exact_match():
    hypotheses = ["She goes to college.", "He has two brothers.", "Wrong answer."]
    references = ["She goes to college.", "He has two brothers.", "Correct answer."]

    em = GECMetrics.compute_exact_match(hypotheses, references)
    assert round(em, 1) == 66.7


def test_compute_f_beta():
    # 1 correct edit, 0 false positives, 0 false negatives
    sources = ["She go to college."]
    hypotheses = ["She goes to college."]
    references = ["She goes to college."]

    res = GECMetrics.compute_f_beta(sources, hypotheses, references, beta=0.5)
    assert res["true_positives"] == 1
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 0
    assert res["precision"] == 100.0
    assert res["recall"] == 100.0
    assert res["f_beta"] == 100.0
