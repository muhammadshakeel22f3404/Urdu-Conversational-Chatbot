import os
import random
import json
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import get_arg_config
from .dataset import load_and_split, UrduChatDataset, collate_fn
from .vocab import Vocab, SOS, EOS
from .tokenizer import detokenize
from .models.transformer import TransformerSeq2Seq
from .utils.metrics import compute_bleu
from .inference import greedy_decode_batch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_vocab_from_pairs(pairs: List, min_freq: int):
    tok_texts = [src for src, _ in pairs] + [tgt for _, tgt in pairs]
    vocab = Vocab(min_freq=min_freq)
    vocab.build(tok_texts)
    return vocab

def save_checkpoint(path, model, vocab, cfg, best_bleu):
    state = {
        "model_state": model.state_dict(),
        "vocab_itos": vocab.itos,
        "vocab_stoi": vocab.stoi,
        "config": cfg.__dict__,
        "best_bleu": best_bleu,
    }
    torch.save(state, path)

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt

def make_dataloaders(train_pairs, val_pairs, vocab, cfg):
    train_ds = UrduChatDataset(train_pairs, vocab, cfg.max_len)
    val_ds = UrduChatDataset(val_pairs, vocab, cfg.max_len)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, vocab))
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, vocab))
    return train_dl, val_dl

def run_validation_bleu(model, dl, vocab, device, subset=500, max_len=64):
    model.eval()
    preds = []
    refs = []
    count = 0
    with torch.no_grad():
        for src, tgt in dl:
            src = src.to(device)
            # reference: remove SOS and EOS
            for t in tgt.tolist():
                # strip <sos> and everything after <eos>
                if len(t) == 0:
                    continue
                # find eos
                try:
                    eos_idx = t.index(vocab.eos_idx)
                except ValueError:
                    eos_idx = len(t)
                ref_ids = t[1:eos_idx]  # skip sos
                refs.append(detokenize(vocab.decode(ref_ids)))
            # predictions
            batch_preds = greedy_decode_batch(model, src, vocab, max_len=max_len, device=device)
            preds.extend(batch_preds)
            count += src.size(0)
            if subset and count >= subset:
                break
    return compute_bleu(preds[:subset], refs[:subset])

def train_epoch(model, dl, vocab, device, optim, criterion):
    model.train()
    total_loss = 0.0
    for src, tgt in tqdm(dl, desc="Train", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        # Decoder input: remove last token; Target output: remove first token
        dec_in = tgt[:, :-1]
        gold = tgt[:, 1:].contiguous()
        logits = model(src, dec_in)  # (B, T-1, V)
        B, T, V = logits.shape
        loss = criterion(logits.view(B*T, V), gold.view(-1))
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += loss.item()
    return total_loss / len(dl)

def main():
    cfg = get_arg_config()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print("Loading and splitting data...")
    train_pairs, val_pairs, test_pairs = load_and_split(cfg.data_path, cfg.input_col, cfg.target_col, seed=cfg.seed)

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    print("Building vocabulary...")
    vocab = build_vocab_from_pairs(train_pairs, cfg.min_freq)
    print(f"Vocab size: {len(vocab.itos)}")

    print("Creating dataloaders...")
    train_dl, val_dl = make_dataloaders(train_pairs, val_pairs, vocab, cfg)

    print("Building model...")
    model = TransformerSeq2Seq(
        src_vocab=len(vocab.itos),
        tgt_vocab=len(vocab.itos),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_enc_layers=cfg.num_enc_layers,
        num_dec_layers=cfg.num_dec_layers,
        d_ff=cfg.d_model*4,
        dropout=cfg.dropout,
        pad_idx=vocab.pad_idx,
        tie_embeddings=cfg.tie_embeddings
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_bleu = -1.0
    best_path = os.path.join(cfg.save_dir, "best_bleu.pt")

    for epoch in range(1, cfg.epochs + 1):
        print(f"Epoch {epoch}/{cfg.epochs}")
        train_loss = train_epoch(model, train_dl, vocab, device, optim, criterion)
        print(f"Train loss: {train_loss:.4f}")

        val_bleu = run_validation_bleu(model, val_dl, vocab, device, subset=cfg.val_bleu_subset, max_len=cfg.max_len)
        print(f"Validation BLEU (subset={cfg.val_bleu_subset}): {val_bleu:.2f}")

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(best_path, model, vocab, cfg, best_bleu)
            print(f"Saved best model to {best_path} with BLEU {best_bleu:.2f}")

    print("Training complete.")
    print(f"Best validation BLEU: {best_bleu:.2f}")

if __name__ == "__main__":
    main()