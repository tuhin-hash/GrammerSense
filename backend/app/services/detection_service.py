"""
Error detection and classification module.
Distinguishes between DIRECTLY_IDENTIFIED (verified via morphological/grammatical rules)
and INFERRED (derived from neural model alignment and contextual fluency).
"""

import re
from typing import List, Dict, Any, Optional
from backend.app.preprocessing.span_aligner import EditSpan


class DetectedError:
    """Represents an error span with precise localization and classification tier."""

    def __init__(
        self,
        error_id: str,
        start_char: int,
        end_char: int,
        original_text: str,
        replacement_text: str,
        category: str,
        verification_tier: str,  # "DIRECTLY_IDENTIFIED" or "INFERRED"
        confidence: float,
        explanation: str,
        rule_id: Optional[str] = None,
    ):
        self.error_id = error_id
        self.start_char = start_char
        self.end_char = end_char
        self.original_text = original_text
        self.replacement_text = replacement_text
        self.category = category
        self.verification_tier = verification_tier
        self.confidence = confidence
        self.explanation = explanation
        self.rule_id = rule_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "original_text": self.original_text,
            "replacement_text": self.replacement_text,
            "category": self.category,
            "verification_tier": self.verification_tier,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "rule_id": self.rule_id,
        }

    def __repr__(self) -> str:
        return (
            f"DetectedError({self.category} [{self.verification_tier}], "
            f"[{self.start_char}:{self.end_char}], orig={self.original_text!r} -> {self.replacement_text!r})"
        )


class DetectionService:
    """Classifies localized edits into validated linguistic categories and tiers."""

    # Lexical tables for verified rule matching
    PLURAL_PRONOUNS = {"they", "we", "you"}
    SINGULAR_PRONOUNS_3SG = {"he", "she", "it"}

    COMMON_PREPOSITIONS = {
        "in", "on", "at", "to", "for", "with", "from", "by", "about", "into",
        "through", "during", "before", "after", "above", "below", "between", "under", "than"
    }

    ARTICLES = {"a", "an", "the"}

    @classmethod
    def classify_edit(
        cls,
        edit: EditSpan,
        source_sentence: str,
        target_sentence: str,
        index: int
    ) -> DetectedError:
        """
        Classifies an EditSpan, determining whether the error is directly identified
        by syntactic/morphological rules or inferred by neural model divergence.
        """
        orig = edit.source_text.strip()
        repl = edit.target_text.strip()
        orig_lower = orig.lower()
        repl_lower = repl.lower()

        # Surrounding context extraction
        prefix = source_sentence[:edit.source_start_char].strip().split()
        prev_word = prefix[-1].lower() if prefix else ""

        error_id = f"err-{index + 1:03d}"

        # -------------------------------------------------------------
        # Tier 1: DIRECTLY IDENTIFIED RULES
        # -------------------------------------------------------------

        # 1. Punctuation & Capitalization
        if orig and repl and orig_lower == repl_lower:
            return DetectedError(
                error_id=error_id,
                start_char=edit.source_start_char,
                end_char=edit.source_end_char,
                original_text=orig,
                replacement_text=repl,
                category="Punctuation & Capitalization",
                verification_tier="DIRECTLY_IDENTIFIED",
                confidence=0.99,
                explanation=f"Capitalization adjustment: change '{orig}' to '{repl}'.",
                rule_id="RULE_CAPITALIZATION"
            )

        if not orig and repl in {".", ",", ";", ":", "!", "?"}:
            return DetectedError(
                error_id=error_id,
                start_char=edit.source_start_char,
                end_char=edit.source_end_char,
                original_text=orig,
                replacement_text=repl,
                category="Punctuation & Capitalization",
                verification_tier="DIRECTLY_IDENTIFIED",
                confidence=0.95,
                explanation=f"Missing punctuation mark '{repl}' at the specified position.",
                rule_id="RULE_PUNCTUATION_INSERT"
            )

        # 2. Subject-Verb Agreement (SVA)
        # e.g., "They was" -> "They were"
        if prev_word in cls.PLURAL_PRONOUNS and orig_lower == "was" and repl_lower == "were":
            return DetectedError(
                error_id=error_id,
                start_char=edit.source_start_char,
                end_char=edit.source_end_char,
                original_text=orig,
                replacement_text=repl,
                category="Subject-Verb Agreement",
                verification_tier="DIRECTLY_IDENTIFIED",
                confidence=0.98,
                explanation=(
                    f"Plural subject '{prev_word}' requires the plural past-tense verb 'were', "
                    f"not singular '{orig}'."
                ),
                rule_id="RULE_SVA_PLURAL_PAST"
            )

        # e.g., "He have" -> "He has"
        if prev_word in cls.SINGULAR_PRONOUNS_3SG and orig_lower == "have" and repl_lower == "has":
            return DetectedError(
                error_id=error_id,
                start_char=edit.source_start_char,
                end_char=edit.source_end_char,
                original_text=orig,
                replacement_text=repl,
                category="Subject-Verb Agreement",
                verification_tier="DIRECTLY_IDENTIFIED",
                confidence=0.98,
                explanation=(
                    f"Third-person singular subject '{prev_word}' requires singular verb 'has', "
                    f"not base form '{orig}'."
                ),
                rule_id="RULE_SVA_3SG_HAVE"
            )

        # e.g., "She go" -> "She goes", "He run" -> "He runs", "She don't" -> "She doesn't"
        if prev_word in cls.SINGULAR_PRONOUNS_3SG:
            if (
                (orig_lower + "s" == repl_lower or orig_lower + "es" == repl_lower)
                or (orig_lower == "don't" and repl_lower == "doesn't")
                or (orig_lower == "do" and repl_lower == "does")
            ):
                return DetectedError(
                    error_id=error_id,
                    start_char=edit.source_start_char,
                    end_char=edit.source_end_char,
                    original_text=orig,
                    replacement_text=repl,
                    category="Subject-Verb Agreement",
                    verification_tier="DIRECTLY_IDENTIFIED",
                    confidence=0.97,
                    explanation=(
                        f"Third-person singular subject '{prev_word}' requires the 3rd-person singular "
                        f"inflected verb '{repl}', not '{orig}'."
                    ),
                    rule_id="RULE_SVA_3SG_INFLECTION"
                )

        # e.g., "The news are" -> "The news is" / "There is many" -> "There are many"
        if orig_lower in {"is", "are"} and repl_lower in {"is", "are"}:
            return DetectedError(
                error_id=error_id,
                start_char=edit.source_start_char,
                end_char=edit.source_end_char,
                original_text=orig,
                replacement_text=repl,
                category="Subject-Verb Agreement",
                verification_tier="DIRECTLY_IDENTIFIED",
                confidence=0.92,
                explanation=(
                    f"Subject-verb agreement: verb number changed from '{orig}' to '{repl}' "
                    f"to agree with the head noun or grammatical subject."
                ),
                rule_id="RULE_SVA_COPULA"
            )

        # 3. Verb Tense & Form
        # e.g., "have went" -> "have gone" / "did not saw" -> "did not see"
        if prev_word in {"have", "has", "had"} and orig_lower == "went" and repl_lower == "gone":
            return DetectedError(
                error_id=error_id,
                start_char=edit.source_start_char,
                end_char=edit.source_end_char,
                original_text=orig,
                replacement_text=repl,
                category="Verb Tense & Form",
                verification_tier="DIRECTLY_IDENTIFIED",
                confidence=0.98,
                explanation="The perfect aspect auxiliary ('have/has/had') requires the past participle 'gone', not past simple 'went'.",
                rule_id="RULE_VERB_PERFECT_PARTICIPLE"
            )

        if prev_word in {"did", "not", "didn't"} and orig_lower in {"saw", "went", "came", "bought"}:
            return DetectedError(
                error_id=error_id,
                start_char=edit.source_start_char,
                end_char=edit.source_end_char,
                original_text=orig,
                replacement_text=repl,
                category="Verb Tense & Form",
                verification_tier="DIRECTLY_IDENTIFIED",
                confidence=0.96,
                explanation="The auxiliary 'did' / 'did not' must be followed by the base bare infinitive form.",
                rule_id="RULE_VERB_DO_SUPPORT"
            )

        # 4. Article / Determiner
        if (orig_lower in cls.ARTICLES and repl_lower in cls.ARTICLES) or (
            edit.operation == "INSERT" and repl_lower in cls.ARTICLES
        ) or (
            edit.operation == "DELETE" and orig_lower in cls.ARTICLES
        ):
            rule_detail = f"Article usage: '{orig}' replaced with '{repl}'." if orig and repl else (
                f"Missing article '{repl}' inserted." if repl else f"Extraneous article '{orig}' removed."
            )
            return DetectedError(
                error_id=error_id,
                start_char=edit.source_start_char,
                end_char=edit.source_end_char,
                original_text=orig,
                replacement_text=repl,
                category="Article / Determiner",
                verification_tier="DIRECTLY_IDENTIFIED",
                confidence=0.94,
                explanation=rule_detail,
                rule_id="RULE_ARTICLE_USAGE"
            )

        # 5. Preposition
        if orig_lower in cls.COMMON_PREPOSITIONS and repl_lower in cls.COMMON_PREPOSITIONS:
            return DetectedError(
                error_id=error_id,
                start_char=edit.source_start_char,
                end_char=edit.source_end_char,
                original_text=orig,
                replacement_text=repl,
                category="Preposition",
                verification_tier="DIRECTLY_IDENTIFIED",
                confidence=0.91,
                explanation=f"Preposition selection: corrected preposition from '{orig}' to '{repl}'.",
                rule_id="RULE_PREPOSITION_SELECTION"
            )

        # 6. Spelling / Typo (Edit distance <= 2 for words >= 3 chars)
        if orig and repl and len(orig) >= 3 and len(repl) >= 3:
            dist = cls._levenshtein_distance(orig_lower, repl_lower)
            if 0 < dist <= 2:
                return DetectedError(
                    error_id=error_id,
                    start_char=edit.source_start_char,
                    end_char=edit.source_end_char,
                    original_text=orig,
                    replacement_text=repl,
                    category="Spelling / Typo",
                    verification_tier="DIRECTLY_IDENTIFIED",
                    confidence=0.89,
                    explanation=f"Possible typo or spelling error: '{orig}' corrected to '{repl}' (edit distance: {dist}).",
                    rule_id="RULE_SPELLING_TYPO"
                )

        # -------------------------------------------------------------
        # Tier 2: INFERRED CLASSIFICATIONS (Neural / Statistical Divergence)
        # -------------------------------------------------------------
        # When no exact rule triggers, classify as INFERRED with contextual explanation
        inferred_category = "Grammatical Structure"
        if edit.operation == "DELETE":
            inferred_category = "Redundancy"
            explanation = f"Extraneous word/phrase '{orig}' removed for grammatical fluency."
        elif edit.operation == "INSERT":
            inferred_category = "Missing Element"
            explanation = f"Suggested addition '{repl}' to complete grammatical structure."
        else:
            inferred_category = "Word Choice / Phrasing"
            explanation = (
                f"Model inferred phrasing adjustment: replaced '{orig}' with '{repl}' "
                f"to improve naturalness and fluency."
            )

        return DetectedError(
            error_id=error_id,
            start_char=edit.source_start_char,
            end_char=edit.source_end_char,
            original_text=orig,
            replacement_text=repl,
            category=inferred_category,
            verification_tier="INFERRED",
            confidence=0.75,
            explanation=explanation,
            rule_id="INFERRED_NEURAL_DIVERGENCE"
        )

    @classmethod
    def detect_and_classify(
        cls,
        source_text: str,
        target_text: str,
        edit_spans: List[EditSpan]
    ) -> List[DetectedError]:
        """Runs classification over all localized edit spans in a sentence."""
        detected: List[DetectedError] = []
        for idx, edit in enumerate(edit_spans):
            err = cls.classify_edit(edit, source_text, target_text, idx)
            detected.append(err)
        return detected

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Computes Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return DetectionService._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
