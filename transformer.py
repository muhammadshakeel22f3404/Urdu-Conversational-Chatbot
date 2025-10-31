import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        B, Tq, _ = q.shape
        B, Tk, _ = k.shape

        q = self.w_q(q).view(B, Tq, self.num_heads, self.d_k).transpose(1, 2)  # (B,h,Tq,d_k)
        k = self.w_k(k).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)  # (B,h,Tk,d_k)
        v = self.w_v(v).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)  # (B,h,Tk,d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,h,Tq,Tk)

        if mask is not None:
            # mask should be broadcastable to (B,h,Tq,Tk); True means "allowed"
            if mask.dtype != torch.bool:
                mask = mask != 0
            # Expand mask heads if needed
            if mask.dim() == 4 and mask.size(1) == 1 and self.num_heads > 1:
                mask = mask.expand(B, self.num_heads, mask.size(2), mask.size(3))
            scores = scores.masked_fill(~mask, float("-inf"))
            # Prevent NaNs if a whole row is masked (rare; e.g., malformed masks)
            all_masked = torch.all(~mask, dim=-1, keepdim=True)  # (B,h,Tq,1)
            scores = scores.masked_fill(all_masked, 0.0)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (B,h,Tq,d_k)
        context = context.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        out = self.w_o(context)
        if return_attn:
            return out, attn
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)

def subsequent_mask(size: int, device=None):
    # (1, size, size) lower-triangular True (allowed), strictly upper False (blocked)
    seq_mask = torch.triu(torch.ones((1, size, size), device=device, dtype=torch.bool), diagonal=1)
    return ~seq_mask  # True where allowed

def make_pad_mask(q_len: int, k_pad_mask: torch.Tensor):
    """
    Create a key padding mask broadcast over queries.
    k_pad_mask: (B, Tk) with True for real tokens and False for pads.
    Returns: (B, 1, Tq, Tk) with True where attention is allowed.
    """
    if k_pad_mask.dtype != torch.bool:
        k_pad_mask = k_pad_mask != 0
    return k_pad_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, q_len, -1)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # Pre-LN
        x2 = self.norm1(x)
        attn = self.self_attn(x2, x2, x2, src_mask)
        x = x + self.dropout(attn)
        x2 = self.norm2(x)
        ff = self.ff(x2)
        x = x + self.dropout(ff)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, src_tgt_mask, return_cross_attn=False):
        x2 = self.norm1(x)
        x_sa = self.self_attn(x2, x2, x2, tgt_mask)
        x = x + self.dropout(x_sa)
        x2 = self.norm2(x)
        if return_cross_attn:
            x_ca, attn = self.cross_attn(x2, memory, memory, src_tgt_mask, return_attn=True)
        else:
            x_ca = self.cross_attn(x2, memory, memory, src_tgt_mask)
            attn = None
        x = x + self.dropout(x_ca)
        x2 = self.norm3(x)
        x_ff = self.ff(x2)
        x = x + self.dropout(x_ff)
        return (x, attn) if return_cross_attn else x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
    def forward(self, src, src_mask):
        x = self.embed(src)  # (B, Ts, D)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, tgt, memory, tgt_mask, src_tgt_mask, return_last_cross_attn=False):
        x = self.embed(tgt)
        x = self.pos(x)
        last_attn = None
        for i, layer in enumerate(self.layers):
            if return_last_cross_attn and i == len(self.layers) - 1:
                x, last_attn = layer(x, memory, tgt_mask, src_tgt_mask, return_cross_attn=True)
            else:
                x = layer(x, memory, tgt_mask, src_tgt_mask)
        x = self.norm(x)
        return (x, last_attn) if return_last_cross_attn else x

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, n_heads, num_enc_layers, num_dec_layers, d_ff, dropout, pad_idx, tie_embeddings=True):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_enc_layers, n_heads, d_ff, dropout, pad_idx)
        self.decoder = Decoder(tgt_vocab, d_model, num_dec_layers, n_heads, d_ff, dropout, pad_idx)
        self.generator = nn.Linear(d_model, tgt_vocab, bias=False)
        if tie_embeddings:
            self.generator.weight = self.decoder.embed.weight

        self.pad_idx = pad_idx
        self.d_model = d_model
        self.n_heads = n_heads

    def make_masks(self, src, tgt):
        # src: (B, Ts), tgt: (B, Tt)
        device = src.device
        # True where token exists (not pad)
        src_key_padding = (src != self.pad_idx)  # (B, Ts), bool
        tgt_key_padding = (tgt != self.pad_idx)  # (B, Tt), bool

        Ts = src.size(1)
        Tt = tgt.size(1)

        # Encoder self-attn: mask keys only (do NOT mask queries)
        src_mask = make_pad_mask(Ts, src_key_padding)  # (B,1,Ts,Ts), True where allowed

        # Decoder self-attn: key padding AND causal mask
        causal = subsequent_mask(Tt, device=device).unsqueeze(1)  # (1,1,Tt,Tt)
        tgt_mask = make_pad_mask(Tt, tgt_key_padding) & causal  # (B,1,Tt,Tt)

        # Cross-attn: mask only encoder keys (source padding)
        src_tgt_mask = make_pad_mask(Tt, src_key_padding)  # (B,1,Tt,Ts)

        return src_mask, tgt_mask, src_tgt_mask

    def forward(self, src, tgt, return_last_cross_attn=False):
        src_mask, tgt_mask, src_tgt_mask = self.make_masks(src, tgt)
        memory = self.encoder(src, src_mask)
        if return_last_cross_attn:
            dec_out, attn = self.decoder(tgt, memory, tgt_mask, src_tgt_mask, return_last_cross_attn=True)
            logits = self.generator(dec_out)
            return logits, attn
        else:
            dec_out = self.decoder(tgt, memory, tgt_mask, src_tgt_mask)
            logits = self.generator(dec_out)
            return logits