import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import load_and_split, UrduChatDataset, collate_fn
from .vocab import Vocab
from .tokenizer import detokenize
from .models.transformer import TransformerSeq2Seq
from .utils.metrics import compute_bleu, compute_chrf, compute_rouge_l, perplexity_from_loss
import torch.nn as nn

def load_ckpt(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    return ckpt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--input_col", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading data...")
    train_pairs, val_pairs, test_pairs = load_and_split(args.data_path, args.input_col, args.target_col)
    print(f"Test size: {len(test_pairs)}")

    print("Loading checkpoint...")
    ckpt = load_ckpt(args.checkpoint, device)
    vocab = Vocab()
    vocab.itos = ckpt["vocab_itos"]
    vocab.stoi = ckpt["vocab_stoi"]

    cfg = ckpt["config"]
    model = TransformerSeq2Seq(
        src_vocab=len(vocab.itos),
        tgt_vocab=len(vocab.itos),
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        num_enc_layers=cfg["num_enc_layers"],
        num_dec_layers=cfg["num_dec_layers"],
        d_ff=cfg["d_model"]*4,
        dropout=cfg["dropout"],
        pad_idx=vocab.pad_idx,
        tie_embeddings=cfg.get("tie_embeddings", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = UrduChatDataset(test_pairs, vocab, args.max_len)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=lambda b: collate_fn(b, vocab))

    preds = []
    refs = []
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    total_loss = 0.0
    steps = 0

    from .inference import greedy_decode_batch
    with torch.no_grad():
        for src, tgt in tqdm(test_dl, desc="Eval"):
            src = src.to(device)
            # loss
            dec_in = tgt[:, :-1].to(device)
            gold = tgt[:, 1:].to(device)
            logits = model(src, dec_in)
            B, T, V = logits.shape
            loss = criterion(logits.view(B*T, V), gold.view(-1))
            total_loss += loss.item()
            steps += 1

            # refs
            for t in tgt.tolist():
                try:
                    eos_idx = t.index(vocab.eos_idx)
                except ValueError:
                    eos_idx = len(t)
                ref_ids = t[1:eos_idx]
                refs.append(detokenize(vocab.decode(ref_ids)))

            # preds
            batch_preds = greedy_decode_batch(model, src, vocab, max_len=args.max_len, device=device)
            preds.extend(batch_preds)

    bleu = compute_bleu(preds, refs)
    chrf = compute_chrf(preds, refs)
    rouge_l = compute_rouge_l(preds, refs)
    ppl = perplexity_from_loss(total_loss / max(1, steps))

    print("=== Test Metrics ===")
    print(f"BLEU:     {bleu:.2f}")
    print(f"chrF:     {chrf:.2f}")
    print(f"ROUGE-L:  {rouge_l:.2f}")
    print(f"Perplexity: {ppl:.2f}")

    # Show a few qualitative examples
    print("\nQualitative examples:")
    for i in range(min(5, len(preds))):
        print(f"- Pred: {preds[i]}")
        print(f"  Ref:  {refs[i]}")

if __name__ == "__main__":
    main()