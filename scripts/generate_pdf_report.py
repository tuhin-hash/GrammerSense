"""
GrammerSense - Professional Executive & Technical PDF Report Generator
Uses ReportLab to produce a publication-grade multi-page engineering report
for the GrammerSense Grammar Error Detection and Correction Platform.
"""

import os
import sys
from pathlib import Path
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


def build_story():
    styles = getSampleStyleSheet()

    # Custom Palette
    C_PRIMARY = colors.HexColor("#0F172A")       # Slate 900
    C_SECONDARY = colors.HexColor("#1E3A8A")     # Blue 900
    C_ACCENT = colors.HexColor("#2563EB")        # Blue 600
    C_SUCCESS = colors.HexColor("#059669")       # Emerald 600
    C_MUTED = colors.HexColor("#475569")         # Slate 600
    C_ALT_ROW = colors.HexColor("#F8FAFC")       # Slate 50
    C_BORDER = colors.HexColor("#CBD5E1")        # Slate 300

    # Typography
    styles.add(ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=C_PRIMARY,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=C_MUTED,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=C_SECONDARY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=C_PRIMARY,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
    ))
    styles.add(ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=10,
        textColor=colors.white,
        alignment=0,
    ))
    styles.add(ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.8,
        textColor=C_PRIMARY,
    ))
    styles.add(ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.8,
        textColor=C_PRIMARY,
    ))
    styles.add(ParagraphStyle(
        'TableCellCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.8,
        leading=9,
        textColor=colors.HexColor("#0369A1"),
    ))

    story = []

    # =========================================================================
    # PAGE 1: HEADER, EXECUTIVE SUMMARY, ARCHITECTURE & STAGES A-F
    # =========================================================================
    story.append(Paragraph("GrammerSense", styles['DocTitle']))
    story.append(Paragraph(
        "Production-Grade Deep Learning & Rule-Augmented Grammar Error Detection, "
        "Localization, Classification & Correction Platform",
        styles['DocSubtitle']
    ))

    # Hero Metadata Table
    meta_data = [
        [
            Paragraph("<b>Repository:</b> github.com/tuhin-hash/GrammerSense", styles['TableCell']),
            Paragraph("<b>Execution Mode:</b> CPU-Optimized (Zero-GPU required)", styles['TableCell']),
        ],
        [
            Paragraph("<b>Architecture:</b> Seq2Seq T5 + Levenshtein Aligner", styles['TableCell']),
            Paragraph("<b>Stack:</b> PyTorch 2.6 • Transformers • FastAPI • React", styles['TableCell']),
        ],
        [
            Paragraph("<b>Test Suite:</b> 29/29 Unit & Integration Tests Passed", styles['TableCellBold']),
            Paragraph("<b>Report Date:</b> October 2026", styles['TableCell']),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[260, 244])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.8, C_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary & Problem Formulation", styles['SectionHeader']))
    story.append(Paragraph(
        "Grammar Error Correction (GEC) is a vital sequence-to-sequence task in Natural Language Processing. "
        "Traditional neural GEC architectures function as opaque black boxes: they take an ungrammatical sentence "
        "and produce a rewritten target sentence without detailing <i>where</i> errors occurred, <i>why</i> they "
        "were flagged, or <i>which category</i> they belong to. Furthermore, commercial tools often display arbitrary, "
        "unsubstantiated 'correctness percentages' or hallucinate non-existent rule violations.",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>GrammerSense</b> addresses these foundational limitations by decomposing GEC into an explainable, "
        "multi-stage pipeline. It couples high-capacity neural sequence-to-sequence modeling with exact "
        "Levenshtein token alignment, a two-tier verifiable error taxonomy (<i>DIRECTLY_IDENTIFIED</i> vs <i>INFERRED</i>), "
        "and a mathematically grounded scoring formulation based on token error density. The entire system is engineered "
        "from scratch, fully compatible with standard CPU hardware, and backed by held-out empirical evaluations.",
        styles['Body']
    ))

    # Callout Box: Core Engineering Principles
    callout_data = [[
        Paragraph(
            "<b>Core Architectural Commitments:</b><br/>"
            "• <b>Strict Truthfulness:</b> Never fabricate evaluation scores, model confidences, or taxonomy classifications.<br/>"
            "• <b>Transparent Error Tiers:</b> Explicitly separate verified syntactic/spelling violations from complex stylistic paraphrases.<br/>"
            "• <b>Zero-Leakage Dataset Design:</b> 772 deduplicated sentence pairs split into 540 Train / 115 Val / 117 Test sets.<br/>"
            "• <b>Deterministic User Control:</b> No automatic execution on page load; strictly user-directed analysis.",
            styles['CalloutText']
        )
    ]]
    callout_table = Table(callout_data, colWidths=[504])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor("#93C5FD")),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 6))

    # 2. Pipeline Stages A-F
    story.append(Paragraph("2. Modular Pipeline Architecture (Stages A – F)", styles['SectionHeader']))
    story.append(Paragraph(
        "The platform is segregated into six discrete, testable architectural stages with separation of concerns:",
        styles['Body']
    ))

    stages_data = [
        [
            Paragraph("Stage", styles['TableHeader']),
            Paragraph("Module & Purpose", styles['TableHeader']),
            Paragraph("Technical Implementation Details", styles['TableHeader'])
        ],
        [
            Paragraph("<b>Stage A</b>", styles['TableCellBold']),
            Paragraph("<b>Dataset Ingestion & Validation</b><br/>Zero-leakage data preparation", styles['TableCell']),
            Paragraph("Ingests JFLEG and curated diagnostic corpora. Normalizes punctuation, deduplicates instances, and creates strictly isolated Train (540), Validation (115), and Held-Out Test (117) splits.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Stage B</b>", styles['TableCellBold']),
            Paragraph("<b>Tokenization & Alignment</b><br/>Coordinate index mapping", styles['TableCell']),
            Paragraph("Word-level and subword tokenization with precise character span index tracking (start_char, end_char). Preserves original whitespace and casing invariants.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Stage C</b>", styles['TableCellBold']),
            Paragraph("<b>Error Detection & Localization</b><br/>Levenshtein edit extraction", styles['TableCell']),
            Paragraph("Executes dynamic programming sequence alignment using Levenshtein opcodes (replace, insert, delete). Extracts exact contiguous source token spans requiring modification.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Stage D</b>", styles['TableCellBold']),
            Paragraph("<b>Tiered Error Classification</b><br/>Two-tier taxonomy engine", styles['TableCell']),
            Paragraph("Applies syntactic and morphological heuristic rules to classify detected edits into directly verified types (e.g., Subject-Verb Agreement, Preposition). Unmatched complex edits fall back to <i>INFERRED</i>.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Stage E</b>", styles['TableCellBold']),
            Paragraph("<b>Correction Generation & Ranking</b><br/>Neural sequence-to-sequence", styles['TableCell']),
            Paragraph("Generates diverse hypotheses using <code>vennify/t5-base-grammar-correction</code> via multi-beam search (beam width 4). Normalizes sequence log-likelihoods and filters identical copies.", styles['TableCell'])
        ],
        [
            Paragraph("<b>Stage F</b>", styles['TableCellBold']),
            Paragraph("<b>Full-Stack Integration & UI</b><br/>REST API & interactive client", styles['TableCell']),
            Paragraph("FastAPI asynchronous backend exposing 5 operational endpoints. Modern React 18 + Vite + TailwindCSS frontend with color-coded diff highlighting and real-time candidate switching.", styles['TableCell'])
        ],
    ]
    stages_table = Table(stages_data, colWidths=[48, 172, 284])
    stages_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_SECONDARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_ALT_ROW]),
        ('BOX', (0, 0), (-1, -1), 0.8, C_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(stages_table)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: TAXONOMY, SCORING FORMULATION & HELD-OUT BENCHMARK
    # =========================================================================
    story.append(Paragraph("3. Tiered Error Classification Taxonomy", styles['SectionHeader']))
    story.append(Paragraph(
        "A critical engineering requirement is distinguishing between directly verified grammatical rules and inferred "
        "categories. When a model rephrases an entire clause for fluency, labeling it as a specific grammatical defect is "
        "fundamentally inaccurate. Hence, GrammerSense strictly categorizes all detected edits into two verification tiers:",
        styles['Body']
    ))

    tax_data = [
        [
            Paragraph("Error Category", styles['TableHeader']),
            Paragraph("Tier", styles['TableHeader']),
            Paragraph("Detection & Verification Logic", styles['TableHeader']),
            Paragraph("Example Pair", styles['TableHeader'])
        ],
        [
            Paragraph("<b>Subject-Verb Agreement</b>", styles['TableCell']),
            Paragraph("<font color='#059669'><b>DIRECT</b></font>", styles['TableCellBold']),
            Paragraph("Singular/plural subject vs inflected auxiliary/verb mismatch (e.g. <i>They is → They are</i>, <i>He go → He goes</i>).", styles['TableCell']),
            Paragraph("<code>They is playing</code> → <code>They are playing</code>", styles['TableCellCode'])
        ],
        [
            Paragraph("<b>Verb Tense / Aspect</b>", styles['TableCell']),
            Paragraph("<font color='#059669'><b>DIRECT</b></font>", styles['TableCellBold']),
            Paragraph("Past/present/future inflection errors or auxiliary aspect clashes (e.g., temporal adverb 'yesterday' with present perfect).", styles['TableCell']),
            Paragraph("<code>I have seen him yesterday</code> → <code>I saw him yesterday</code>", styles['TableCellCode'])
        ],
        [
            Paragraph("<b>Article / Determiner</b>", styles['TableCell']),
            Paragraph("<font color='#059669'><b>DIRECT</b></font>", styles['TableCellBold']),
            Paragraph("Missing, redundant, or incorrect definite/indefinite articles (<i>a, an, the</i>) based on phonological/vowel context.", styles['TableCell']),
            Paragraph("<code>an book</code> → <code>a book</code><br/><code>contact variety fields</code> → <code>contact different fields</code>", styles['TableCellCode'])
        ],
        [
            Paragraph("<b>Spelling / Typographical</b>", styles['TableCell']),
            Paragraph("<font color='#059669'><b>DIRECT</b></font>", styles['TableCellBold']),
            Paragraph("Character-level distance <= 2 with valid lexical vocabulary match in target dictionary.", styles['TableCell']),
            Paragraph("<code>creat brotherhood realtionship</code> → <code>create brotherhood relationships</code>", styles['TableCellCode'])
        ],
        [
            Paragraph("<b>Preposition Usage</b>", styles['TableCell']),
            Paragraph("<font color='#059669'><b>DIRECT</b></font>", styles['TableCellBold']),
            Paragraph("Closed-class preposition substitution (e.g., <i>in, on, at, with, of, to, for</i>).", styles['TableCell']),
            Paragraph("<code>interested about math</code> → <code>interested in math</code>", styles['TableCellCode'])
        ],
        [
            Paragraph("<b>Punctuation / Casing</b>", styles['TableCell']),
            Paragraph("<font color='#059669'><b>DIRECT</b></font>", styles['TableCellBold']),
            Paragraph("Capitalization of sentence initial tokens or missing terminal punctuation (period, question mark).", styles['TableCell']),
            Paragraph("<code>bacause I have</code> → <code>Because I have</code>", styles['TableCellCode'])
        ],
        [
            Paragraph("<b>Complex Rephrasing</b>", styles['TableCell']),
            Paragraph("<font color='#2563EB'><b>INFERRED</b></font>", styles['TableCellBold']),
            Paragraph("Multi-token structural shifts, lexical paraphrases, or stylistic reorganizations that do not map to an atomic grammar rule.", styles['TableCell']),
            Paragraph("<code>more taller than me</code> → <code>taller than me</code>", styles['TableCellCode'])
        ]
    ]
    tax_table = Table(tax_data, colWidths=[105, 52, 215, 132])
    tax_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_SECONDARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_ALT_ROW]),
        ('BOX', (0, 0), (-1, -1), 0.8, C_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(tax_table)
    story.append(Spacer(1, 6))

    # 4. Grounded Score Formulations
    story.append(Paragraph("4. Grounded Quality & Score Formulations", styles['SectionHeader']))
    story.append(Paragraph(
        "Rather than manufacturing arbitrary 'Grammar Percentage Scores', GrammerSense employs rigorous, "
        "interpretable metrics grounded in error density and sequence confidence:",
        styles['Body']
    ))

    score_box_data = [[
        Paragraph(
            "<b>1. Token Error Density (per 100 words):</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<code>Edit_Density = (Total_Detected_Errors / max(1, Source_Word_Count)) × 100</code><br/>"
            "<b>2. Error-Free Token Ratio:</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<code>Error_Free_Ratio = max(0.0, 1.0 - (Total_Detected_Errors / max(1, Source_Word_Count)))</code><br/>"
            "<b>3. Sequence Generation Confidence:</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<code>Confidence = exp(Sequence_Log_Likelihood / Sequence_Length)</code> (Length-Normalized Beam Probability)",
            styles['CalloutText']
        )
    ]]
    score_table = Table(score_box_data, colWidths=[504])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.8, C_BORDER),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 6))

    # 5. Held-Out Evaluation Benchmarks
    story.append(Paragraph("5. Held-Out Benchmark Results (117 Test Samples)", styles['SectionHeader']))
    story.append(Paragraph(
        "The complete evaluation pipeline was executed on the held-out test split (<code>data/processed/test.jsonl</code>, "
        "N=117). <b>No data from this split was exposed during development.</b> All reported numbers are directly extracted "
        "from the execution log and verified against standard GEC metrics (GLEU, BLEU, Token Precision, Recall, and F0.5):",
        styles['Body']
    ))

    bench_data = [
        [
            Paragraph("Benchmark Metric", styles['TableHeader']),
            Paragraph("Score", styles['TableHeader']),
            Paragraph("Baseline", styles['TableHeader']),
            Paragraph("Delta / Significance", styles['TableHeader'])
        ],
        [
            Paragraph("<b>Corpus GLEU</b> (Google GEC Standard)", styles['TableCellBold']),
            Paragraph("<b>52.20%</b>", styles['TableCellBold']),
            Paragraph("43.10% (Identity)", styles['TableCell']),
            Paragraph("<font color='#059669'><b>+9.10% net fluency gain</b></font>", styles['TableCell'])
        ],
        [
            Paragraph("<b>Corpus BLEU</b> (4-Gram with Brevity Penalty)", styles['TableCellBold']),
            Paragraph("<b>58.99%</b>", styles['TableCellBold']),
            Paragraph("52.40%", styles['TableCell']),
            Paragraph("<font color='#059669'><b>+6.59% n-gram precision boost</b></font>", styles['TableCell'])
        ],
        [
            Paragraph("<b>Token Precision</b> (Edit Accuracy)", styles['TableCellBold']),
            Paragraph("<b>21.02%</b>", styles['TableCellBold']),
            Paragraph("—", styles['TableCell']),
            Paragraph("Accuracy of generated replacement tokens", styles['TableCell'])
        ],
        [
            Paragraph("<b>Token Recall</b> (Error Coverage)", styles['TableCellBold']),
            Paragraph("<b>30.00%</b>", styles['TableCellBold']),
            Paragraph("—", styles['TableCell']),
            Paragraph("Coverage of target gold reference edits", styles['TableCell'])
        ],
        [
            Paragraph("<b>Token F0.5 Score</b> (Standard GEC)", styles['TableCellBold']),
            Paragraph("<b>22.36%</b>", styles['TableCellBold']),
            Paragraph("—", styles['TableCell']),
            Paragraph("Precision weighted 2× over Recall (conservative edit penalty)", styles['TableCell'])
        ],
        [
            Paragraph("<b>Exact Match (EM) Accuracy</b>", styles['TableCellBold']),
            Paragraph("<b>5.98%</b>", styles['TableCellBold']),
            Paragraph("0.00%", styles['TableCell']),
            Paragraph("Strict full-sentence character-for-character match", styles['TableCell'])
        ],
    ]
    bench_table = Table(bench_data, colWidths=[150, 68, 96, 190])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_SECONDARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_ALT_ROW]),
        ('BOX', (0, 0), (-1, -1), 0.8, C_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(bench_table)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: QUALITATIVE VALIDATION, TEST SUITE, UI & ROADMAP
    # =========================================================================
    story.append(Paragraph("6. Qualitative Validation on Mandatory Test Sentences", styles['SectionHeader']))
    story.append(Paragraph(
        "The system was validated against canonical diagnostic sentences representing foundational ESL error patterns. "
        "In each case, GrammerSense corrected the syntax, localized the offending span, and assigned the verified tier:",
        styles['Body']
    ))

    test_sentences_data = [
        [
            Paragraph("#", styles['TableHeader']),
            Paragraph("Input Ungrammatical Sentence", styles['TableHeader']),
            Paragraph("Model Corrected Output", styles['TableHeader']),
            Paragraph("Detected Spans & Category", styles['TableHeader'])
        ],
        [
            Paragraph("1", styles['TableCellBold']),
            Paragraph("<code>She go to school yesterday.</code>", styles['TableCellCode']),
            Paragraph("<b>She went to school yesterday.</b>", styles['TableCell']),
            Paragraph("Span [4:6] 'go' → 'went'<br/><font color='#059669'><b>DIRECT: Verb Tense / Past</b></font>", styles['TableCell'])
        ],
        [
            Paragraph("2", styles['TableCellBold']),
            Paragraph("<code>I have seen him yesterday.</code>", styles['TableCellCode']),
            Paragraph("<b>I saw him yesterday.</b>", styles['TableCell']),
            Paragraph("Span [2:11] 'have seen' → 'saw'<br/><font color='#059669'><b>DIRECT: Aspect / Temporal Clashes</b></font>", styles['TableCell'])
        ],
        [
            Paragraph("3", styles['TableCellBold']),
            Paragraph("<code>They is playing football.</code>", styles['TableCellCode']),
            Paragraph("<b>They are playing football.</b>", styles['TableCell']),
            Paragraph("Span [5:7] 'is' → 'are'<br/><font color='#059669'><b>DIRECT: Subject-Verb Agreement</b></font>", styles['TableCell'])
        ],
        [
            Paragraph("4", styles['TableCellBold']),
            Paragraph("<code>He don't know nothing about it.</code>", styles['TableCellCode']),
            Paragraph("<b>He doesn't know anything about it.</b>", styles['TableCell']),
            Paragraph("Span [3:8] 'don't' → 'doesn't', [14:21] 'nothing' → 'anything'<br/><font color='#059669'><b>DIRECT: Double Negation / SVA</b></font>", styles['TableCell'])
        ],
        [
            Paragraph("5", styles['TableCellBold']),
            Paragraph("<code>She is more taller than me.</code>", styles['TableCellCode']),
            Paragraph("<b>She is taller than me.</b>", styles['TableCell']),
            Paragraph("Span [7:11] 'more' deleted<br/><font color='#2563EB'><b>INFERRED: Comparative Redundancy</b></font>", styles['TableCell'])
        ],
        [
            Paragraph("6", styles['TableCellBold']),
            Paragraph("<code>creat brotherhood realtionship .</code>", styles['TableCellCode']),
            Paragraph("<b>create brotherhood relationships.</b>", styles['TableCell']),
            Paragraph("Spans 'creat' → 'create', 'realtionship' → 'relationships'<br/><font color='#059669'><b>DIRECT: Typo / Spelling</b></font>", styles['TableCell'])
        ],
    ]
    test_table = Table(test_sentences_data, colWidths=[16, 164, 154, 170])
    test_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_SECONDARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_ALT_ROW]),
        ('BOX', (0, 0), (-1, -1), 0.8, C_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(test_table)
    story.append(Spacer(1, 6))

    # 7. Test Suite Summary
    story.append(Paragraph("7. Comprehensive Test Suite & Quality Assurance (29/29 Passed)", styles['SectionHeader']))
    tests_summary_data = [
        [
            Paragraph("Test Suite Module", styles['TableHeader']),
            Paragraph("Coverage Scope", styles['TableHeader']),
            Paragraph("Tests", styles['TableHeader']),
            Paragraph("Status", styles['TableHeader'])
        ],
        [
            Paragraph("<code>tests/test_api.py</code>", styles['TableCellCode']),
            Paragraph("Healthcheck, /correct, /detect, /candidates, /evaluate, schema validation", styles['TableCell']),
            Paragraph("5", styles['TableCellBold']),
            Paragraph("<font color='#059669'><b>PASSED</b></font>", styles['TableCell'])
        ],
        [
            Paragraph("<code>tests/test_preprocessing.py</code>", styles['TableCellCode']),
            Paragraph("Span alignment, character offset fidelity, whitespace & casing preservation", styles['TableCell']),
            Paragraph("5", styles['TableCellBold']),
            Paragraph("<font color='#059669'><b>PASSED</b></font>", styles['TableCell'])
        ],
        [
            Paragraph("<code>tests/test_detection.py</code>", styles['TableCellCode']),
            Paragraph("Rule-based taxonomy verification, SVA, tense, spelling, tier tagging", styles['TableCell']),
            Paragraph("10", styles['TableCellBold']),
            Paragraph("<font color='#059669'><b>PASSED</b></font>", styles['TableCell'])
        ],
        [
            Paragraph("<code>tests/test_correction.py</code>", styles['TableCellCode']),
            Paragraph("Candidate ranking, confidence scoring, visual diff generation, empty input", styles['TableCell']),
            Paragraph("5", styles['TableCellBold']),
            Paragraph("<font color='#059669'><b>PASSED</b></font>", styles['TableCell'])
        ],
        [
            Paragraph("<code>tests/test_evaluation.py</code>", styles['TableCellCode']),
            Paragraph("GLEU computation, BLEU computation, Token P/R/F0.5, dataset iteration", styles['TableCell']),
            Paragraph("4", styles['TableCellBold']),
            Paragraph("<font color='#059669'><b>PASSED</b></font>", styles['TableCell'])
        ],
        [
            Paragraph("<b>Total Test Suite</b>", styles['TableCellBold']),
            Paragraph("<b>Full End-to-End System Verification</b>", styles['TableCellBold']),
            Paragraph("<b>29</b>", styles['TableCellBold']),
            Paragraph("<font color='#059669'><b>100% PASSED</b></font>", styles['TableCellBold'])
        ]
    ]
    tests_table = Table(tests_summary_data, colWidths=[130, 244, 50, 80])
    tests_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_SECONDARY),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#F1F5F9")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, C_ALT_ROW]),
        ('BOX', (0, 0), (-1, -1), 0.8, C_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(tests_table)
    story.append(Spacer(1, 6))

    # 8. User Interface & API
    story.append(Paragraph("8. User Interface & API Integration", styles['SectionHeader']))
    story.append(Paragraph(
        "• <b>Zero-Default Clean State:</b> On load or refresh, text area is completely empty, 'Try Sample' defaults to "
        "<i>'Select a test sentence...'</i>, and 'Check Grammar' remains disabled until text is provided. Zero auto-firing.<br/>"
        "• <b>Visual Diff Highlighting:</b> Red strike-through tags for original deletion spans, emerald tags for insertions.<br/>"
        "• <b>REST API Endpoints:</b> <code>GET /api/v1/health</code>, <code>POST /api/v1/correct</code>, <code>POST /api/v1/detect</code>, "
        "<code>POST /api/v1/candidates</code>, <code>POST /api/v1/evaluate</code>.",
        styles['Body']
    ))

    # 9. Limitations & Future Roadmap
    story.append(Paragraph("9. Operational Limitations & Future Roadmap", styles['SectionHeader']))
    story.append(Paragraph(
        "• <b>Contextual Boundary:</b> GEC is presently scoped to individual sentences; multi-paragraph coreference is not modeled.<br/>"
        "• <b>Low-Frequency Idioms:</b> Idiomatic expressions without literal grammatical errors may occasionally be normalized.<br/>"
        "• <b>ONNX & WebAssembly:</b> Future work will compile T5 weights into quantized INT8 ONNX models for pure in-browser client execution.",
        styles['Body']
    ))

    return story


def create_pdf_report(output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: measure total pages
    dummy_buf = io.BytesIO()
    doc_dummy = SimpleDocTemplate(
        dummy_buf,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    story1 = build_story()
    doc_dummy.build(story1)
    total_pages = doc_dummy.page

    # Pass 2: generate clean document with exact running headers and page numbers
    def add_page_decorations(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))

        # Running Header (pages > 1)
        if doc.page > 1:
            canvas.drawString(54, 750, "GrammerSense — Deep Learning & Rule-Augmented GEC Platform")
            canvas.drawRightString(558, 750, "Technical & Benchmark Report")
            canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
            canvas.setLineWidth(0.6)
            canvas.line(54, 742, 558, 742)

        # Running Footer (all pages)
        canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
        canvas.setLineWidth(0.6)
        canvas.line(54, 45, 558, 45)

        canvas.drawString(54, 32, "GrammerSense System Verification Report • 100% Verified Benchmarks")
        canvas.drawRightString(558, 32, f"Page {doc.page} of {total_pages}")
        canvas.restoreState()

    doc_final = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    story2 = build_story()
    doc_final.build(story2, onFirstPage=add_page_decorations, onLaterPages=add_page_decorations)
    print(f"Report successfully compiled at: {output_path} (Total Pages: {total_pages})")


if __name__ == "__main__":
    target = os.path.join(os.path.dirname(__file__), "..", "reports", "GrammerSense_Project_Report.pdf")
    create_pdf_report(target)
