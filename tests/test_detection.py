"""
Unit tests for error detection, localization, and classification.
Specifically tests mandatory user benchmark sentences, directly identified vs inferred errors,
and already-correct sentences.
"""

import pytest
from backend.app.preprocessing.span_aligner import SpanAligner
from backend.app.services.detection_service import DetectionService


def test_mandatory_example_1_she_go():
    src = "She go to college every day."
    tgt = "She goes to college every day."
    edits = SpanAligner.align(src, tgt)
    detected = DetectionService.detect_and_classify(src, tgt, edits)

    assert len(detected) == 1
    err = detected[0]
    assert err.category == "Subject-Verb Agreement"
    assert err.verification_tier == "DIRECTLY_IDENTIFIED"
    assert err.original_text == "go"
    assert err.replacement_text == "goes"
    assert "She" in err.explanation or "singular" in err.explanation.lower()


def test_mandatory_example_2_he_have():
    src = "He have two brothers."
    tgt = "He has two brothers."
    edits = SpanAligner.align(src, tgt)
    detected = DetectionService.detect_and_classify(src, tgt, edits)

    assert len(detected) == 1
    err = detected[0]
    assert err.category == "Subject-Verb Agreement"
    assert err.verification_tier == "DIRECTLY_IDENTIFIED"
    assert err.original_text == "have"
    assert err.replacement_text == "has"


def test_mandatory_example_3_they_was():
    src = "They was playing cricket."
    tgt = "They were playing cricket."
    edits = SpanAligner.align(src, tgt)
    detected = DetectionService.detect_and_classify(src, tgt, edits)

    assert len(detected) == 1
    err = detected[0]
    assert err.category == "Subject-Verb Agreement"
    assert err.verification_tier == "DIRECTLY_IDENTIFIED"
    assert err.original_text == "was"
    assert err.replacement_text == "were"


def test_mandatory_example_4_correct_market():
    src = "I went to the market yesterday."
    tgt = "I went to the market yesterday."
    edits = SpanAligner.align(src, tgt)
    detected = DetectionService.detect_and_classify(src, tgt, edits)

    assert len(detected) == 0


def test_mandatory_example_5_correct_student():
    src = "She is a good student."
    tgt = "She is a good student."
    edits = SpanAligner.align(src, tgt)
    detected = DetectionService.detect_and_classify(src, tgt, edits)

    assert len(detected) == 0


def test_verb_tense_detection():
    src = "I have went to London."
    tgt = "I have gone to London."
    edits = SpanAligner.align(src, tgt)
    detected = DetectionService.detect_and_classify(src, tgt, edits)

    assert len(detected) == 1
    err = detected[0]
    assert err.category == "Verb Tense & Form"
    assert err.verification_tier == "DIRECTLY_IDENTIFIED"
    assert err.original_text == "went"
    assert err.replacement_text == "gone"


def test_article_detection():
    src = "This is an useful tool."
    tgt = "This is a useful tool."
    edits = SpanAligner.align(src, tgt)
    detected = DetectionService.detect_and_classify(src, tgt, edits)

    assert len(detected) == 1
    err = detected[0]
    assert err.category == "Article / Determiner"
    assert err.verification_tier == "DIRECTLY_IDENTIFIED"


def test_preposition_detection():
    src = "He is interested on art."
    tgt = "He is interested in art."
    edits = SpanAligner.align(src, tgt)
    detected = DetectionService.detect_and_classify(src, tgt, edits)

    assert len(detected) == 1
    err = detected[0]
    assert err.category == "Preposition"
    assert err.verification_tier == "DIRECTLY_IDENTIFIED"


def test_inferred_classification():
    src = "She suggested me to visit doctor."
    tgt = "She suggested that I visit the doctor."
    edits = SpanAligner.align(src, tgt)
    detected = DetectionService.detect_and_classify(src, tgt, edits)

    assert len(detected) >= 1
    # Check that complex structural edit is appropriately classified as INFERRED
    inferred_errors = [e for e in detected if e.verification_tier == "INFERRED"]
    assert len(inferred_errors) >= 1
