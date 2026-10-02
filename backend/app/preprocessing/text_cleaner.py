"""
Text preprocessing and tokenization module.
Preserves character offsets for accurate error localization in original input strings.
"""

import re
from typing import List, Tuple, Dict, Any


class TokenSpan:
    def __init__(self, text: str, start: int, end: int, index: int):
        self.text = text
        self.start = start
        self.end = end
        self.index = index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "index": self.index
        }

    def __repr__(self) -> str:
        return f"TokenSpan({self.text!r}, [{self.start}:{self.end}], #{self.index})"


class TextCleaner:
    """Handles text normalization, sentence boundary detection, and offset-preserving tokenization."""

    # Regex for tokenization that separates words and punctuation while preserving offsets
    TOKEN_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """Trims leading/trailing whitespace and collapses internal repeated spaces."""
        if not text:
            return ""
        return " ".join(text.split())

    @classmethod
    def split_sentences(cls, text: str) -> List[Tuple[str, int, int]]:
        """
        Splits text into sentences while recording start and end character offsets.
        Returns list of (sentence_text, start_char, end_char).
        """
        if not text:
            return []

        # Regex matching sentence boundaries considering common punctuation followed by space/end
        sentence_end_pattern = re.compile(r"([^.!?]+[.!?]*|\S+)", re.UNICODE)
        sentences: List[Tuple[str, int, int]] = []

        # Find spans using regex search
        for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text):
            span_text = match.group(0)
            cleaned = span_text.strip()
            if cleaned:
                # Find start index of trimmed content within span
                rel_start = span_text.find(cleaned)
                start = match.start() + rel_start
                end = start + len(cleaned)
                sentences.append((cleaned, start, end))

        if not sentences and text.strip():
            sentences.append((text.strip(), 0, len(text.strip())))

        return sentences

    @classmethod
    def tokenize_with_offsets(cls, text: str) -> List[TokenSpan]:
        """
        Tokenizes text into words and punctuation marks, preserving exact character offsets.
        """
        tokens: List[TokenSpan] = []
        if not text:
            return tokens

        for idx, match in enumerate(cls.TOKEN_REGEX.finditer(text)):
            tok_text = match.group(0)
            start = match.start()
            end = match.end()
            tokens.append(TokenSpan(text=tok_text, start=start, end=end, index=idx))

        return tokens

    @classmethod
    def tokens_to_strings(cls, tokens: List[TokenSpan]) -> List[str]:
        """Extracts list of raw token strings."""
        return [t.text for t in tokens]
