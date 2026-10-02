"""
Data preparation, cleaning, deduplication, and split generation.
Enforces strict SHA-256 deduplication to guarantee ZERO data leakage between train, validation, and test sets.
"""

import json
import hashlib
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DEMO_DIR = BASE_DIR / "data" / "demo"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RANDOM_SEED = 42


def get_sentence_hash(text: str) -> str:
    """Returns SHA-256 hex digest of normalized sentence string."""
    normalized = " ".join(text.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_pairs() -> List[Dict[str, Any]]:
    """Loads pairs from raw and demo datasets."""
    pairs: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()

    sources_to_check = [
        RAW_DIR / "benchmark_raw_pairs.jsonl",
        DEMO_DIR / "demo_pairs.jsonl",
    ]

    for path in sources_to_check:
        if not path.exists():
            continue
        logger.info(f"Loading data from {path.name}...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    src = item.get("source", "").strip()
                    tgt = item.get("target", "").strip()
                    if not src or not tgt:
                        continue

                    shash = get_sentence_hash(src)
                    if shash not in seen_hashes:
                        seen_hashes.add(shash)
                        pairs.append({
                            "id": item.get("id", f"pair-{len(pairs)+1:04d}"),
                            "source": src,
                            "target": tgt,
                            "error_type": item.get("error_type", "General"),
                            "verification_tier": item.get("verification_tier", "INFERRED"),
                            "source_hash": shash,
                        })
                except json.JSONDecodeError:
                    continue

    logger.info(f"Loaded {len(pairs)} unique sentence pairs after strict SHA-256 deduplication.")
    return pairs


def prepare_and_split(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Partitions pairs into non-overlapping train, validation, and test splits."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6

    pairs = load_pairs()
    if not pairs:
        logger.error("No valid pairs found to prepare. Run download_data.py first.")
        return

    # Shuffle deterministically
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    
    train_data = pairs[:n_train]
    val_data = pairs[n_train : n_train + n_val]
    test_data = pairs[n_train + n_val :]

    # Safety: Verify zero leakage
    train_hashes = {p["source_hash"] for p in train_data}
    val_hashes = {p["source_hash"] for p in val_data}
    test_hashes = {p["source_hash"] for p in test_data}

    leak_train_val = train_hashes.intersection(val_hashes)
    leak_train_test = train_hashes.intersection(test_hashes)
    leak_val_test = val_hashes.intersection(test_hashes)

    if leak_train_val or leak_train_test or leak_val_test:
        raise RuntimeError("DATA LEAKAGE DETECTED! Overlapping hashes between splits.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def write_split(path: Path, data: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Wrote {len(data)} items to {path.name}")

    write_split(PROCESSED_DIR / "train.jsonl", train_data)
    write_split(PROCESSED_DIR / "val.jsonl", val_data)
    write_split(PROCESSED_DIR / "test.jsonl", test_data)

    summary = {
        "random_seed": RANDOM_SEED,
        "total_unique_pairs": n,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "data_leakage_verified_zero": True,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio
        }
    }

    with open(PROCESSED_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Dataset preparation complete with zero data leakage.")


if __name__ == "__main__":
    prepare_and_split()
