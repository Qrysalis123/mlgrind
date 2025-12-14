"""
Data preparation and high-throughput training loader.
Usage:
    # Prepare data ONCE (tokenize + chunk + save)
    python data.py
    # Training scripts:
    from data import get_dataloader
"""
import os
from itertools import chain
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# -------------------------------------------------------------------
# PREPROCESS + SAVE TO DISK
# -------------------------------------------------------------------
def prepare_data(
    dataset_name="HuggingFaceFW/fineweb-edu",
    subset="sample-10BT",
    split="train",
    seq_len=1024,
    output_dir=".cache/fineweb-edu-10BT",
    num_proc=8,
):
    """Download, tokenize, chunk, and save dataset to disk."""
    print(f"\n▶ Loading dataset: {dataset_name} ({subset})")
    data = load_dataset(dataset_name, name=subset, split=split)
    tokenizer = AutoTokenizer.from_pretrained("bytebpe-tokenizer-32k-mlm-uncased", use_fast=True)
    # ---------------------------------------------------------------
    # Tokenize step
    # ---------------------------------------------------------------
    def tokenize(examples):
        text = examples.get("text") or examples.get("ctx")
        if text is None:
            raise ValueError(f"Unknown text column: {examples.keys()}")
        # Only return input_ids
        return tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding=False,
        )
    print("▶ Tokenizing...")
    tokenized = data.map(
        tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=data.column_names,
        desc="Tokenizing",
    )
    del data
    # ---------------------------------------------------------------
    # Chunk into fixed seq_len sequences
    # ---------------------------------------------------------------
    def group_texts(examples):
        concatenated = list(chain(*examples["input_ids"]))
        total_length = (len(concatenated) // seq_len) * seq_len
        return {
            "input_ids": [
                concatenated[i : i + seq_len]
                for i in range(0, total_length, seq_len)
            ]
        }
    print(f"▶ Chunking sequences into {seq_len}-token blocks...")
    chunked = tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc=f"Chunking into {seq_len}-token blocks",
    )
    del tokenized
    # ---------------------------------------------------------------
    # Save to disk
    # ---------------------------------------------------------------
    print(f"▶ Saving {len(chunked):,} sequences to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    chunked.save_to_disk(output_dir)
    print("✔ Done.\n")

# -------------------------------------------------------------------
# TRAINING DATALOADER — FAST + SHUFFLE
# -------------------------------------------------------------------
def get_dataloader(
    data_dir,
    batch_size,
    distributed=False,
    num_workers=4,
):
    """
    Load preprocessed dataset from disk and build a shuffle-able DataLoader.
    """
    print(f"▶ Loading preprocessed dataset from {data_dir}")
    dataset = load_from_disk(data_dir)
    dataset.set_format(type="torch", columns=["input_ids"])
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),  # shuffle only if not using DDP
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    output_dir = ".cache/fineweb-edu-10BT"
    if os.path.exists(output_dir):
        print(f"✔ Dataset already prepared at {output_dir} — skipping preprocessing.")
    else:
        prepare_data(
            dataset_name="HuggingFaceFW/fineweb-edu",
            subset="sample-10BT",
            split="train",
            seq_len=1024,
            output_dir=output_dir,
            num_proc=8,
        )
