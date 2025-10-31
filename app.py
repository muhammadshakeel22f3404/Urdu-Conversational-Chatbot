import argparse
import torch
import gradio as gr
from .vocab import Vocab
from .models.transformer import TransformerSeq2Seq
from .tokenizer import tokenize, detokenize
from .utils.urdu_normalize import normalize_urdu
from .inference import greedy_decode, beam_search_decode

CSS = """
/* Force RTL in text elements */
* { direction: rtl; text-align: right; }
"""

def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
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
    return model, vocab, cfg

def respond(message, history, decoding, max_len, beam_size, model, vocab, device):
    # Optionally include history: concat previous pairs to add context (simple heuristic)
    # We will only use current message for simplicity
    msg = normalize_urdu(message)
    src_tokens = tokenize(msg)
    src_ids = [vocab.stoi.get(t, vocab.unk_idx) for t in (["<sos>"] + src_tokens + ["<eos>"])]
    src_tensor = torch.tensor(src_ids, dtype=torch.long)

    if decoding == "Greedy":
        out_ids = greedy_decode(model, src_tensor, vocab, max_len=max_len, device=device)
    else:
        out_ids = beam_search_decode(model, src_tensor, vocab, beam_size=beam_size, max_len=max_len, device=device)

    reply = detokenize(vocab.decode(out_ids))
    history = history + [(message, reply)]
    return history, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, cfg = load_model(args.checkpoint, device)

    with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown("## اردو چیٹ بوٹ (Transformer from scratch)")
        chatbot = gr.Chatbot(label="گفتگو")
        with gr.Row():
            msg = gr.Textbox(label="پیغام لکھیں", placeholder="سلام! آپ کیسے ہیں؟")
        with gr.Row():
            decoding = gr.Radio(["Greedy", "Beam"], value="Greedy", label="Decoding")
            max_len = gr.Slider(16, 128, value=cfg.get("gen_max_len", 64), step=1, label="Max Length")
            beam_size = gr.Slider(2, 8, value=cfg.get("beam_size", 4), step=1, label="Beam Size")
        clear = gr.Button("صاف کریں")
        submit = gr.Button("ارسال کریں")

        state = gr.State([])

        def on_submit(message, history, decoding, max_len, beam_size):
            return respond(message, history, decoding, int(max_len), int(beam_size), model, vocab, device)

        submit.click(on_submit, [msg, state, decoding, max_len, beam_size], [chatbot, state])
        msg.submit(on_submit, [msg, state, decoding, max_len, beam_size], [chatbot, state])
        clear.click(lambda: ([], []), None, [chatbot, state])

    demo.launch(share=args.share)

if __name__ == "__main__":
    main()