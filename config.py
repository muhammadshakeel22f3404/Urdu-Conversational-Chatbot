import argparse
from dataclasses import dataclass, asdict

@dataclass
class Config:
    data_path: str = "/content/drive/MyDrive/urdu_chatbot/data/final_main_dataset.tsv"
    input_col: str = "input"
    target_col: str = "response"
    min_freq: int = 2
    max_len: int = 64
    batch_size: int = 64
    d_model: int = 256
    n_heads: int = 2
    num_enc_layers: int = 2
    num_dec_layers: int = 2
    d_ff: int = 1024
    dropout: float = 0.1
    lr: float = 3e-4
    epochs: int = 30
    device: str = "cuda"
    save_dir: str = "checkpoints"
    val_bleu_subset: int = 500  # compute BLEU on subset for speed
    seed: int = 42
    tie_embeddings: bool = True

    # inference defaults
    beam_size: int = 4
    gen_max_len: int = 64

def get_arg_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=Config.data_path)
    parser.add_argument("--input_col", type=str, default=Config.input_col)
    parser.add_argument("--target_col", type=str, default=Config.target_col)
    parser.add_argument("--min_freq", type=int, default=Config.min_freq)
    parser.add_argument("--max_len", type=int, default=Config.max_len)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--d_model", type=int, default=Config.d_model)
    parser.add_argument("--n_heads", type=int, default=Config.n_heads)
    parser.add_argument("--num_enc_layers", type=int, default=Config.num_enc_layers)
    parser.add_argument("--num_dec_layers", type=int, default=Config.num_dec_layers)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--save_dir", type=str, default=Config.save_dir)
    parser.add_argument("--val_bleu_subset", type=int, default=Config.val_bleu_subset)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--tie_embeddings", action="store_true", default=Config.tie_embeddings)
    parser.add_argument("--beam_size", type=int, default=Config.beam_size)
    parser.add_argument("--gen_max_len", type=int, default=Config.gen_max_len)
    args = parser.parse_args()
    cfg = Config(**vars(args))
    return cfg