"""
Unit tests for text preprocessing, tokenization, and offset preservation.
"""

import pytest
from backend.app.preprocessing.text_cleaner import TextCleaner
from backend.app.preprocessing.span_aligner import SpanAligner


def test_normalize_whitespace():
    assert TextCleaner.normalize_whitespace("   Hello    world!   ") == "Hello world!"
    assert TextCleaner.normalize_whitespace("") == ""
    assert TextCleaner.normalize_whitespace("One\nTwo\tThree") == "One Two Three"


def test_tokenize_with_offsets():
    text = "She goes to college."
    tokens = TextCleaner.tokenize_with_offsets(text)

    assert len(tokens) == 5
    assert tokens[0].text == "She"
    assert tokens[0].start == 0
    assert tokens[0].end == 3

    assert tokens[1].text == "goes"
    assert tokens[1].start == 4
    assert tokens[1].end == 8

    # Verify character substring matches original
    for t in tokens:
        assert text[t.start:t.end] == t.text


def test_split_sentences():
    text = "She went home. He stayed late! What happened?"
    sentences = TextCleaner.split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0][0] == "She went home."
    assert sentences[1][0] == "He stayed late!"
    assert sentences[2][0] == "What happened?"


def test_span_aligner_substitution():
    src = "She go to college."
    tgt = "She goes to college."
    edits = SpanAligner.align(src, tgt)

    assert len(edits) == 1
    edit = edits[0]
    assert edit.operation == "REPLACE"
    assert edit.source_text == "go"
    assert edit.target_text == "goes"
    assert src[edit.source_start_char : edit.source_end_char] == "go"


def test_span_aligner_identical():
    src = "She is a good student."
    tgt = "She is a good student."
    edits = SpanAligner.align(src, tgt)
    assert len(edits) == 0


def test_span_aligner_insertion():
    src = "She bought apple."
    tgt = "She bought an apple."
    edits = SpanAligner.align(src, tgt)
    assert len(edits) >= 1
    insert_edit = [e for e in edits if e.operation == "INSERT"]
    assert len(insert_edit) == 1
    assert insert_edit[0].target_text == "an"
