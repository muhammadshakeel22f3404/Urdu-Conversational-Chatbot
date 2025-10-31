import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
from .utils.urdu_normalize import normalize_urdu
from .tokenizer import tokenize
from .vocab import Vocab, SOS, EOS

# Common column name pairs to auto-detect
COMMON_COLS = [
    ("input", "response"),
    ("question", "answer"),
    ("prompt", "response"),
    ("user", "bot"),
    ("input_text", "target_text"),
    ("source", "target"),
    ("input", "output"),
    ("sentence", "sentence"),  # added for your dataset structure
]

def autodetect_columns(df: pd.DataFrame, input_col=None, target_col=None) -> Tuple[str, str]:
    """Try to detect input/target column names automatically."""
    cols = [c.lower() for c in df.columns]
    if input_col and target_col:
        return input_col, target_col

    for a, b in COMMON_COLS:
        if a in cols and b in cols:
            return df.columns[cols.index(a)], df.columns[cols.index(b)]
    # fallback: first two columns
    return df.columns[0], df.columns[1]

def preprocess_text(s: str) -> str:
    """Normalize Urdu text by removing diacritics, standardizing characters, etc."""
    return normalize_urdu(str(s))

def tokenize_pair(inp: str, tgt: str) -> Tuple[List[str], List[str]]:
    """Normalize and tokenize a pair of input and target sentences."""
    return tokenize(preprocess_text(inp)), tokenize(preprocess_text(tgt))

class UrduChatDataset(Dataset):
    """Custom Dataset for Urdu conversational chatbot."""
    def __init__(self, pairs: List[Tuple[List[str], List[str]]], vocab: Vocab, max_len: int):
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_toks, tgt_toks = self.pairs[idx]
        src_toks = src_toks[: self.max_len - 2]
        tgt_toks = tgt_toks[: self.max_len - 2]
        src = [SOS] + src_toks + [EOS]
        tgt = [SOS] + tgt_toks + [EOS]
        return src, tgt

def collate_fn(batch, vocab: Vocab):
    """Pad sequences in a batch to same length."""
    src_batch, tgt_batch = zip(*batch)
    src_ids = [vocab.encode(x) for x in src_batch]
    tgt_ids = [vocab.encode(x) for x in tgt_batch]
    src_lens = [len(x) for x in src_ids]
    tgt_lens = [len(x) for x in tgt_ids]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    pad = vocab.pad_idx

    src_tensor = torch.full((len(batch), max_src), pad, dtype=torch.long)
    tgt_tensor = torch.full((len(batch), max_tgt), pad, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_ids, tgt_ids)):
        src_tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_tensor[i, : len(t)] = torch.tensor(t, dtype=torch.long)

    return src_tensor, tgt_tensor

def load_and_split(
    data_path: str,
    input_col: str = None,
    target_col: str = None,
    test_size=0.1,
    val_size=0.1,
    seed=42
):
    """Load dataset, detect columns, normalize, tokenize, and split into train/val/test."""
    assert os.path.exists(data_path), f"File not found: {data_path}"

    print(f"Loading dataset from: {data_path}")

    # --- Detect delimiter automatically ---
    try:
        if data_path.endswith(".tsv"):
            df = pd.read_csv(data_path, sep="\t")
        else:
            # Try comma first, fallback to tab if error occurs
            try:
                df = pd.read_csv(data_path)
            except pd.errors.ParserError:
                print("CSV parsing failed — retrying with tab separator...")
                df = pd.read_csv(data_path, sep="\t")
    except Exception as e:
        raise RuntimeError(f"Failed to read dataset: {e}")

    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    inp_col, tgt_col = autodetect_columns(df, input_col, target_col)
    print(f"Using columns → Input: '{inp_col}' | Target: '{tgt_col}'")

    df = df[[inp_col, tgt_col]].dropna().astype(str)

    # Tokenize each input/target pair
    print("Tokenizing dataset...")
    pairs = [tokenize_pair(i, t) for i, t in zip(df[inp_col].values, df[tgt_col].values)]
    print(f"Tokenized {len(pairs)} sentence pairs")

    # Train/Val/Test split 80/10/10
    train_pairs, tmp_pairs = train_test_split(
        pairs, test_size=val_size + test_size, random_state=seed
    )
    rel_val = val_size / (val_size + test_size)
    val_pairs, test_pairs = train_test_split(tmp_pairs, test_size=1 - rel_val, random_state=seed)

    print(f"Split sizes → Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs