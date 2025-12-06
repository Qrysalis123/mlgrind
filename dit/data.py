import tiktoken
from datasets import load_dataset
from itertools import chain
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_dataloaders(
    train_dataset_name,
    train_split,
    train_subset,
    valid_dataset_name,
    valid_split,
    batch_size,
    seq_len,
    cache_dir=None,
    num_proc=8,
    distributed=True,
    num_workers=4
):
    """
    Load HuggingFace datasets and create train/validation dataloaders.

    Args:
        train_dataset_name: HuggingFace dataset name for training
        train_split: Split to use for training (e.g., 'train')
        train_subset: Subset name for training (e.g., 'sample-10BT')
        valid_dataset_name: HuggingFace dataset name for validation (None to skip)
        valid_split: Split to use for validation (e.g., 'validation')
        batch_size: Batch size per GPU
        seq_len: Sequence length for each sample
        cache_dir: Cache directory for datasets
        num_proc: Number of processes for parallel tokenization
        distributed: Whether to use DistributedSampler for multi-GPU training
        num_workers: Number of dataloader workers

    Returns:
        train_loader, val_loader: PyTorch DataLoaders (val_loader is None if valid_dataset_name is None)
    """
    # Load tiktoken GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Load datasets
    print(f"Loading {train_dataset_name} ({train_subset})...")
    train_data = load_dataset(
        train_dataset_name,
        name=train_subset,
        split=train_split,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Load validation dataset only if specified
    if valid_dataset_name is not None:
        print(f"Loading {valid_dataset_name}...")
        valid_data = load_dataset(
            valid_dataset_name,
            split=valid_split,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    else:
        valid_data = None

    # Tokenization function
    def tokenize(examples):
        # Handle different text column names
        if "text" in examples:
            texts = examples["text"]
        elif "ctx" in examples:  # hellaswag uses 'ctx'
            texts = examples["ctx"]
        else:
            raise ValueError(f"Unknown text column. Available: {list(examples.keys())}")

        # Tokenize using tiktoken
        tokens = [enc.encode(text) for text in texts]
        return {"input_ids": tokens}

    # Tokenize datasets
    print("Tokenizing training data...")
    train_tokenized = train_data.map(
        tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=train_data.column_names,
        desc="Tokenizing train"
    )
    del train_data

    if valid_data is not None:
        print("Tokenizing validation data...")
        valid_tokenized = valid_data.map(
            tokenize,
            batched=True,
            num_proc=num_proc,
            remove_columns=valid_data.column_names,
            desc="Tokenizing valid"
        )
        del valid_data
    else:
        valid_tokenized = None

    # Group texts into chunks
    def group_texts(examples):
        # Concatenate all texts
        concatenated = list(chain(*examples["input_ids"]))
        total_length = len(concatenated)

        # Drop remainder
        total_length = (total_length // seq_len) * seq_len

        # Split into chunks
        result = {
            "input_ids": [
                concatenated[i : i + seq_len]
                for i in range(0, total_length, seq_len)
            ]
        }
        return result

    print("Chunking training data...")
    train_chunked = train_tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Chunking train"
    )
    del train_tokenized
    train_chunked.set_format(type="torch", columns=["input_ids"])

    if valid_tokenized is not None:
        print("Chunking validation data...")
        valid_chunked = valid_tokenized.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            desc="Chunking valid"
        )
        del valid_tokenized
        valid_chunked.set_format(type="torch", columns=["input_ids"])
    else:
        valid_chunked = None

    # Create samplers
    if distributed:
        # DistributedSampler: assigns unique indices to Each GPU
        # each getting a subset of the chunked data (total sequences, Seq_len). non-overlapping
        train_sampler = DistributedSampler(train_chunked, shuffle=True)
    else:
        train_sampler = None

    # Create dataloaders
    # Batches them together into (Batch_num, Seq Len) for each GPU e.g (64, 1024)
    train_loader = DataLoader(
        train_chunked,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    if valid_chunked is not None:
        if distributed:
            valid_sampler = DistributedSampler(valid_chunked, shuffle=False)
        else:
            valid_sampler = None
        valid_loader = DataLoader(
            valid_chunked,
            batch_size=batch_size,
            sampler=valid_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
    else:
        valid_loader = None

    return train_loader, valid_loader
