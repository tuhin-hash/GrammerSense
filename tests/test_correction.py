"""
Unit tests for candidate selection, ranking, and visual diff generation.
"""

import pytest
from backend.app.correction.candidate_selector import CandidateSelector
from backend.app.services.correction_service import CorrectionService


def test_candidate_selector_ranking():
    source = "She go to college."
    raw_candidates = [
        ("She goes to college.", 0.85),
        ("She is going to college.", 0.10),
        ("She go to college.", 0.05),
    ]

    best, all_cands = CandidateSelector.select_best(source, raw_candidates)

    assert best.candidate_text == "She goes to college."
    assert best.is_selected is True
    assert len(all_cands) == 3
    assert all_cands[0].rank == 1
    assert all_cands[1].rank == 2
    assert all_cands[2].rank == 3


def test_candidate_selector_already_correct_stability():
    source = "She is a good student."
    raw_candidates = [
        ("She is a good student.", 0.70),
        ("She was a good student.", 0.30),
    ]

    best, all_cands = CandidateSelector.select_best(source, raw_candidates)
    assert best.candidate_text == "She is a good student."
    assert best.edit_distance == 0


def test_generate_visual_diff():
    source = "She go to college"
    target = "She goes to college"

    diffs = CorrectionService._generate_visual_diff(source, target)
    types = [d["type"] for d in diffs]

    assert "delete" in types
    assert "insert" in types

    del_item = next(d for d in diffs if d["type"] == "delete")
    ins_item = next(d for d in diffs if d["type"] == "insert")

    assert del_item["text"] == "go"
    assert ins_item["text"] == "goes"


def test_generate_visual_diff_identical():
    source = "She is a good student"
    target = "She is a good student"

    diffs = CorrectionService._generate_visual_diff(source, target)
    assert len(diffs) == 1
    assert diffs[0]["type"] == "equal"
    assert diffs[0]["text"] == source
