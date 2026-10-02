"""
Fine-tuning pipeline for Sequence-to-Sequence Grammar Error Correction.
Supports PyTorch CPU/GPU training, validation loss tracking, and checkpoint saving.
Records complete reproducibility metadata.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"


class GECDataset(Dataset):
    """Dataset for input-target grammar error correction pairs."""

    def __init__(self, data_path: Path, tokenizer, max_length: int = 128, max_samples: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                self.examples.append(item)
                if max_samples and len(self.examples) >= max_samples:
                    break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        src_text = f"grammar: {item['source']}"
        tgt_text = item["target"]

        input_enc = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_enc = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_enc.input_ids.squeeze(0)
        # Replace pad token id with -100 so CrossEntropy ignores them
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "labels": labels,
        }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Training on device: {device}")

    # Set reproducibility seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Loading base model and tokenizer: {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.to(device)

    train_file = args.dataset_dir / "train.jsonl"
    val_file = args.dataset_dir / "val.jsonl"

    if not train_file.exists():
        logger.warning(f"{train_file} not found. Running prepare_data.py first...")
        from scripts.prepare_data import prepare_and_split
        prepare_and_split()

    train_dataset = GECDataset(
        train_file,
        tokenizer,
        max_length=args.max_length,
        max_samples=args.max_train_samples
    )
    val_dataset = GECDataset(
        val_file,
        tokenizer,
        max_length=args.max_length,
        max_samples=args.max_val_samples
    )

    logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []

    logger.info(f"Beginning training for {args.epochs} epoch(s)...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if (step + 1) % args.log_interval == 0 or (step + 1) == len(train_loader):
                logger.info(f"Epoch {epoch}/{args.epochs} | Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / max(1, len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        logger.info(f"=== Epoch {epoch} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ===")

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4)
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"New best validation loss! Saving checkpoint to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    # Save training report
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reproducibility": {
            "base_model": args.model_name_or_path,
            "device": str(device),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        },
        "best_val_loss": round(best_val_loss, 4),
        "training_history": history,
    }

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training pipeline finished successfully. Checkpoint saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Seq2Seq model for Grammar Error Correction.")
    parser.add_argument("--model_name_or_path", type=str, default="t5-small")
    parser.add_argument("--dataset_dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output_dir", type=Path, default=CHECKPOINTS_DIR / "best_model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--dry_run", action="store_true", help="Quick 1-step verification run")

    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 1
        args.max_train_samples = 4
        args.max_val_samples = 2
        args.batch_size = 2
        args.log_interval = 1

    train(args)


if __name__ == "__main__":
    main()
