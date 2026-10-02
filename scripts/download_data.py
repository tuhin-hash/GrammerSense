"""
Data download script for GEC benchmark datasets.
Fetches JFLEG dataset (JHU FLuency-Extended GUG) test and dev splits from public repository,
or initializes benchmark data in data/raw.
"""

import sys
import json
import logging
from pathlib import Path
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEMO_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "demo"

# Official JFLEG dataset raw links
JFLEG_TEST_SRC = "https://raw.githubusercontent.com/keisks/jfleg/master/test/test.src"
JFLEG_TEST_REF0 = "https://raw.githubusercontent.com/keisks/jfleg/master/test/test.ref0"
JFLEG_DEV_SRC = "https://raw.githubusercontent.com/keisks/jfleg/master/dev/dev.src"
JFLEG_DEV_REF0 = "https://raw.githubusercontent.com/keisks/jfleg/master/dev/dev.ref0"


def download_url(url: str, target: Path) -> bool:
    try:
        logger.info(f"Downloading {url} -> {target.name}...")
        urllib.request.urlretrieve(url, target)
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def build_raw_datasets():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    test_src_path = RAW_DATA_DIR / "jfleg_test.src"
    test_ref_path = RAW_DATA_DIR / "jfleg_test.ref0"
    dev_src_path = RAW_DATA_DIR / "jfleg_dev.src"
    dev_ref_path = RAW_DATA_DIR / "jfleg_dev.ref0"

    success_test_src = download_url(JFLEG_TEST_SRC, test_src_path)
    success_test_ref = download_url(JFLEG_TEST_REF0, test_ref_path)
    success_dev_src = download_url(JFLEG_DEV_SRC, dev_src_path)
    success_dev_ref = download_url(JFLEG_DEV_REF0, dev_ref_path)

    pairs_built = 0
    raw_output_file = RAW_DATA_DIR / "benchmark_raw_pairs.jsonl"

    if success_test_src and success_test_ref and test_src_path.exists() and test_ref_path.exists():
        with open(test_src_path, "r", encoding="utf-8") as fs, open(test_ref_path, "r", encoding="utf-8") as fr, open(raw_output_file, "w", encoding="utf-8") as fo:
            for idx, (src_line, ref_line) in enumerate(zip(fs, fr)):
                s = src_line.strip()
                r = ref_line.strip()
                if s and r:
                    entry = {
                        "id": f"jfleg-{idx+1:04d}",
                        "source": s,
                        "target": r,
                        "source_dataset": "JFLEG_test"
                    }
                    fo.write(json.dumps(entry) + "\n")
                    pairs_built += 1
        logger.info(f"Successfully processed {pairs_built} pairs from downloaded JFLEG dataset into {raw_output_file}")
    else:
        logger.info("External download unavailable or skipped; copying demo dataset as baseline raw data...")
        demo_file = DEMO_DATA_DIR / "demo_pairs.jsonl"
        if demo_file.exists():
            import shutil
            shutil.copy(demo_file, raw_output_file)
            logger.info(f"Copied demo dataset to {raw_output_file}")


if __name__ == "__main__":
    build_raw_datasets()
