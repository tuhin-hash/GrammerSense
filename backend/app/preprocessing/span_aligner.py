"""
Sequence alignment and error localization module.
Applies dynamic programming edit distance to map source tokens to target tokens,
localizing edit spans with exact character offsets.
"""

from typing import List, Dict, Any, Optional
import difflib
from .text_cleaner import TextCleaner, TokenSpan


class EditSpan:
    """Represents a localized edit between source and target text."""
    def __init__(
        self,
        operation: str,  # 'REPLACE', 'INSERT', 'DELETE'
        source_start_char: int,
        source_end_char: int,
        source_text: str,
        target_text: str,
        source_token_start: int,
        source_token_end: int,
        target_token_start: int,
        target_token_end: int,
    ):
        self.operation = operation
        self.source_start_char = source_start_char
        self.source_end_char = source_end_char
        self.source_text = source_text
        self.target_text = target_text
        self.source_token_start = source_token_start
        self.source_token_end = source_token_end
        self.target_token_start = target_token_start
        self.target_token_end = target_token_end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "source_start_char": self.source_start_char,
            "source_end_char": self.source_end_char,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "source_token_start": self.source_token_start,
            "source_token_end": self.source_token_end,
            "target_token_start": self.target_token_start,
            "target_token_end": self.target_token_end,
        }

    def __repr__(self) -> str:
        return (
            f"EditSpan({self.operation}, [{self.source_start_char}:{self.source_end_char}], "
            f"orig={self.source_text!r} -> repl={self.target_text!r})"
        )


class SpanAligner:
    """Aligns token sequences and extracts localized character-level edit spans."""

    @classmethod
    def align(cls, source_text: str, target_text: str) -> List[EditSpan]:
        """
        Calculates the edit alignment between source_text and target_text.
        Returns a list of EditSpans containing character boundaries and token indices.
        """
        source_tokens = TextCleaner.tokenize_with_offsets(source_text)
        target_tokens = TextCleaner.tokenize_with_offsets(target_text)

        src_words = [t.text.lower() for t in source_tokens]
        tgt_words = [t.text.lower() for t in target_tokens]

        # Use SequenceMatcher with autojunk disabled for deterministic token alignment
        matcher = difflib.SequenceMatcher(None, src_words, tgt_words, autojunk=False)
        opcodes = matcher.get_opcodes()

        edit_spans: List[EditSpan] = []

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                # Check for case differences in tokens that matched in lower()
                for si, tj in zip(range(i1, i2), range(j1, j2)):
                    src_t = source_tokens[si]
                    tgt_t = target_tokens[tj]
                    if src_t.text != tgt_t.text:
                        edit_spans.append(
                            EditSpan(
                                operation="REPLACE",
                                source_start_char=src_t.start,
                                source_end_char=src_t.end,
                                source_text=src_t.text,
                                target_text=tgt_t.text,
                                source_token_start=si,
                                source_token_end=si + 1,
                                target_token_start=tj,
                                target_token_end=tj + 1,
                            )
                        )
                continue

            if tag == "replace":
                src_sub = source_tokens[i1:i2]
                tgt_sub = target_tokens[j1:j2]
                start_char = src_sub[0].start
                end_char = src_sub[-1].end
                orig_text = source_text[start_char:end_char]
                repl_text = target_text[tgt_sub[0].start:tgt_sub[-1].end]

                edit_spans.append(
                    EditSpan(
                        operation="REPLACE",
                        source_start_char=start_char,
                        source_end_char=end_char,
                        source_text=orig_text,
                        target_text=repl_text,
                        source_token_start=i1,
                        source_token_end=i2,
                        target_token_start=j1,
                        target_token_end=j2,
                    )
                )

            elif tag == "delete":
                src_sub = source_tokens[i1:i2]
                start_char = src_sub[0].start
                end_char = src_sub[-1].end
                orig_text = source_text[start_char:end_char]

                edit_spans.append(
                    EditSpan(
                        operation="DELETE",
                        source_start_char=start_char,
                        source_end_char=end_char,
                        source_text=orig_text,
                        target_text="",
                        source_token_start=i1,
                        source_token_end=i2,
                        target_token_start=j1,
                        target_token_end=j1,
                    )
                )

            elif tag == "insert":
                tgt_sub = target_tokens[j1:j2]
                repl_text = target_text[tgt_sub[0].start:tgt_sub[-1].end]

                # Anchor insertion point to nearest character in source
                if i1 < len(source_tokens):
                    anchor_char = source_tokens[i1].start
                elif len(source_tokens) > 0:
                    anchor_char = source_tokens[-1].end
                else:
                    anchor_char = 0

                edit_spans.append(
                    EditSpan(
                        operation="INSERT",
                        source_start_char=anchor_char,
                        source_end_char=anchor_char,
                        source_text="",
                        target_text=repl_text,
                        source_token_start=i1,
                        source_token_end=i1,
                        target_token_start=j1,
                        target_token_end=j2,
                    )
                )

        return edit_spans
