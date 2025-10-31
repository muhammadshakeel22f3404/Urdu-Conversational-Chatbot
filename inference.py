from typing import List
import torch
import torch.nn.functional as F
from .vocab import SOS, EOS

def _ids_until_eos(ids: List[int], eos_idx: int):
    out = []
    for t in ids:
        if t == eos_idx:
            break
        out.append(t)
    return out

def greedy_decode(model, src, vocab, max_len=64, device="cpu"):
    model.eval()
    with torch.no_grad():
        src = src.unsqueeze(0).to(device)  # (1, Ts)
        # Start with <sos>
        tgt = torch.tensor([[vocab.stoi[SOS]]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            logits = model(src, tgt)  # (1, T, V)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1,1)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == vocab.eos_idx:
                break
        # remove <sos>, stop at EOS
        out_ids = _ids_until_eos(tgt.squeeze(0).tolist()[1:], vocab.eos_idx)
        return out_ids

def greedy_decode_batch(model, src_batch, vocab, max_len=64, device="cpu"):
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(src_batch.size(0)):
            out_ids = greedy_decode(model, src_batch[i], vocab, max_len=max_len, device=device)
            text = " ".join(vocab.decode(out_ids))
            results.append(text)
    return results

def beam_search_decode(model, src, vocab, beam_size=4, max_len=64, device="cpu", length_penalty=0.6):
    model.eval()
    with torch.no_grad():
        src = src.unsqueeze(0).to(device)
        sos = vocab.sos_idx
        eos = vocab.eos_idx

        beams = [(torch.tensor([[sos]], device=device, dtype=torch.long), 0.0)]  # (seq, logprob)
        completed = []

        for _ in range(max_len - 1):
            new_beams = []
            for seq, score in beams:
                if seq[0, -1].item() == eos:
                    completed.append((seq, score))
                    continue
                logits = model(src, seq)  # (1, T, V)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # (1, V)
                topk = torch.topk(log_probs, beam_size, dim=-1)
                for k in range(beam_size):
                    next_id = topk.indices[0, k].view(1, 1)
                    next_score = score + topk.values[0, k].item()
                    new_seq = torch.cat([seq, next_id], dim=1)
                    new_beams.append((new_seq, next_score))
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]
            # if all beams ended
            if all(seq[0, -1].item() == eos for seq, _ in beams):
                completed.extend(beams)
                break

        if not completed:
            completed = beams

        # length penalty
        def lp(len_seq):
            return ((5 + len_seq) / 6) ** length_penalty

        scored = [(seq, score / lp(seq.size(1))) for seq, score in completed]
        best_seq, best_score = max(scored, key=lambda x: x[1])
        ids = best_seq.squeeze(0).tolist()[1:]  # remove SOS
        ids = _ids_until_eos(ids, eos)
        return ids