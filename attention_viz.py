import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from .models.transformer import TransformerSeq2Seq
from .vocab import Vocab
from .tokenizer import detokenize

def visualize_last_cross_attention(model: TransformerSeq2Seq, src, tgt_inp, vocab: Vocab, device="cpu"):
    # Run forward with attention
    model.eval()
    with torch.no_grad():
        logits, attn = model(src.to(device), tgt_inp.to(device), return_last_cross_attn=True)
        # attn: (B, heads, T_tgt, T_src)
        attn = attn[0].mean(0).cpu().numpy()  # average over heads -> (T_tgt, T_src)
        src_tokens = vocab.decode(src[0].cpu().tolist())
        tgt_tokens = vocab.decode(tgt_inp[0].cpu().tolist())
        # Trim at EOS
        if "<eos>" in src_tokens:
            src_tokens = src_tokens[: src_tokens.index("<eos>")+1]
            attn = attn[:, :len(src_tokens)]
        if "<eos>" in tgt_tokens:
            tgt_tokens = tgt_tokens[: tgt_tokens.index("<eos>")+1]
            attn = attn[:len(tgt_tokens), :]

        plt.figure(figsize=(min(12, 0.4*len(src_tokens)), min(8, 0.4*len(tgt_tokens))))
        plt.imshow(attn, aspect="auto", cmap="viridis")
        plt.yticks(np.arange(len(tgt_tokens)), tgt_tokens, fontsize=8)
        plt.xticks(np.arange(len(src_tokens)), src_tokens, rotation=90, fontsize=8)
        plt.colorbar()
        plt.title("Last Decoder Layer Cross-Attention (avg over heads)")
        plt.tight_layout()
        plt.show()