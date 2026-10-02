"""
Interactive and batch CLI inference script for Grammar Error Detection & Correction.
Runs real model inference, span localization, error classification, and candidate ranking.
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.services.correction_service import CorrectionService
from backend.app.models.grammar_model import GrammarModel


def print_result(result: dict):
    print("\n" + "=" * 65)
    print("                GRAMMAR ANALYSIS RESULT")
    print("=" * 65)
    print(f"Original Text  : {result['original_text']}")
    print(f"Corrected Text : {result['corrected_text']}")
    print(f"Detected Errors: {len(result['detected_errors'])}")
    print(f"Confidence     : {result['metrics']['model_confidence']:.2%}")
    print(f"Error-free Tok : {result['metrics']['error_free_token_ratio']:.2%}")

    if result["detected_errors"]:
        print("\n--- Detected Error Details ---")
        for err in result["detected_errors"]:
            tier_badge = f"[{err['verification_tier']}]"
            print(f" * {err['category']} {tier_badge}:")
            print(f"   Position   : characters [{err['start_char']}:{err['end_char']}]")
            print(f"   Change     : '{err['original_text']}' -> '{err['replacement_text']}'")
            print(f"   Confidence : {err['confidence']:.2f}")
            print(f"   Explanation: {err['explanation']}")
    else:
        print("\n[OK] No grammatical errors detected! The text appears grammatically sound.")

    if len(result["candidates"]) > 1:
        print("\n--- Alternative Candidates ---")
        for cand in result["candidates"]:
            sel = " (SELECTED)" if cand["is_selected"] else ""
            print(f" #{cand['rank']}: \"{cand['candidate_text']}\" (Conf: {cand['confidence']:.2%}){sel}")

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="CLI Grammar Error Detection & Correction")
    parser.add_argument("--text", type=str, help="Single input text to correct")
    parser.add_argument("--file", type=Path, help="Input text file containing sentences (one per line)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI prompt")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")
    parser.add_argument("--candidates", type=int, default=3, help="Number of candidates to generate")

    args = parser.parse_args()

    model = GrammarModel.get_instance()
    service = CorrectionService(model=model)

    if args.text:
        res = service.process_text(args.text, num_candidates=args.candidates)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print_result(res)

    elif args.file:
        if not args.file.exists():
            print(f"Error: File not found at {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        print(f"Processing {len(lines)} sentences from {args.file}...\n")
        for line in lines:
            res = service.process_text(line, num_candidates=args.candidates)
            if args.json:
                print(json.dumps(res))
            else:
                print_result(res)

    elif args.interactive or len(sys.argv) == 1:
        print("\n" + "=" * 65)
        print("  Grammar Error Detection & Correction (CLI Interactive Mode)")
        print("  Type a sentence and press Enter. Type 'exit' or 'quit' to stop.")
        print("=" * 65 + "\n")

        while True:
            try:
                user_input = input("Enter sentence > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("Exiting.")
                    break

                res = service.process_text(user_input, num_candidates=args.candidates)
                print_result(res)
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break


if __name__ == "__main__":
    main()
