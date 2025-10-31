# Urdu Conversational Chatbot: Transformer with Multi-Head Attention (From Scratch)

This project implements a custom Urdu conversational chatbot using a Transformer encoder–decoder with multi-head attention in PyTorch. It includes preprocessing (Urdu normalization, tokenization), training with teacher forcing, evaluation (BLEU, ROUGE-L, chrF, Perplexity), and a Gradio UI for real-time interaction.

Dataset (Kaggle):
- https://www.kaggle.com/datasets/muhammadahmedansari/urdu-dataset-20000

Note: You can run this entirely in Google Colab.

## Features
- Transformer (Encoder–Decoder) with Multi-Head Attention, Positional Encoding, Feedforward layers
- Urdu text normalization (remove diacritics, standardize Alef/Yeh, etc.)
- From-scratch tokenizer + vocabulary with special tokens (<pad>, <sos>, <eos>, <unk>)
- Train/Val/Test split (80/10/10)
- Teacher forcing (decoder uses shifted targets)
- Save best model by validation BLEU
- Evaluation: BLEU, ROUGE-L, chrF, Perplexity
- Inference with Greedy and Beam Search
- Gradio UI with Urdu right-to-left rendering support
- Optional attention visualization
- Colab-friendly: step-by-step commands

## Colab: Step-by-step

1) Open Google Colab, set Runtime → Change runtime type → Hardware accelerator → GPU.

2) Create project folders:
```bash
!mkdir -p /content/urdu_chatbot/src/models /content/urdu_chatbot/src/utils /content/urdu_chatbot/checkpoints /content/urdu_chatbot/data
```

3) Install dependencies:
```bash
%cd /content/urdu_chatbot
!pip -q install -r /content/urdu_chatbot/requirements.txt
```

4) Add the files below into Colab:
- For each file shown in this README (requirements.txt, src/*.py, etc.), create a cell with:
```bash
%%writefile /content/urdu_chatbot/<path/to/file>
# paste the file content here (exactly as provided)
```
Do this for all files included below in this README. Alternatively, copy-paste each file block directly in Colab with the `%%writefile` magic.

5) Download the dataset:
Option A: Kaggle API
- Upload your `kaggle.json` to Colab (left sidebar → Files → Upload).
```bash
!pip -q install kaggle
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d muhammadahmedansari/urdu-dataset-20000 -p /content/urdu_chatbot/data
!unzip -o /content/urdu_chatbot/data/urdu-dataset-20000.zip -d /content/urdu_chatbot/data
```
Check the extracted CSV filename in `/content/urdu_chatbot/data`. Update `--data_path` accordingly in the training command.

Option B: Manual Upload
- Download CSV from Kaggle to your computer and upload to `/content/urdu_chatbot/data` as `urdu_data.csv`.

6) Train the model:
```bash
%cd /content/urdu_chatbot
!python -u src/train.py \
  --data_path /content/urdu_chatbot/data/urdu_data.csv \
  --input_col input --target_col response \
  --min_freq 2 \
  --max_len 64 \
  --batch_size 64 \
  --d_model 256 \
  --n_heads 2 \
  --num_enc_layers 2 \
  --num_dec_layers 2 \
  --dropout 0.1 \
  --lr 3e-4 \
  --epochs 10 \
  --val_bleu_subset 500 \
  --save_dir /content/urdu_chatbot/checkpoints
```

Notes:
- If your dataset columns have different names, use `--input_col` and `--target_col` to match. The loader also tries to auto-detect common names (`question/answer`, `prompt/response`, `user/bot`, `input/output`).
- Training on a GPU is recommended.

7) Evaluate:
```bash
!python -u src/evaluate.py \
  --data_path /content/urdu_chatbot/data/urdu_data.csv \
  --input_col input --target_col response \
  --max_len 64 \
  --checkpoint /content/urdu_chatbot/checkpoints/best_bleu.pt
```

8) Run the Gradio UI:
```bash
!python -u src/app.py --checkpoint /content/urdu_chatbot/checkpoints/best_bleu.pt --share
```
- Colab will print a public "gradio.live" URL; click to open the chatbot.
- Choose decoding strategy (Greedy/Beam), adjust max length/beam size, and chat in Urdu. UI supports RTL.

## Tips
- If you get CUDA OOM errors, reduce `batch_size` and/or `max_len`.
- Increase `epochs` for better results.
- Try `d_model=512` for stronger model (use smaller batch size if needed).
- Try different `n_heads`, layers, and dropout for analysis.

## Deliverables
- Code (this repository structure)
- Trained checkpoint saved in `/checkpoints/best_bleu.pt`
- Evaluation metrics printed by `evaluate.py`
- Gradio public link for demo (or deploy on Streamlit Cloud if you port to Streamlit)
- A Medium blog post; suggested structure:
  - Introduction: task, challenges with Urdu, dataset overview
  - Model: Transformer architecture with diagrams and math (attention/positional encoding)
  - Preprocessing: Urdu normalization/tokenization details
  - Training: hyperparameters, teacher forcing, BLEU checkpointing
  - Evaluation: BLEU, ROUGE-L, chrF, Perplexity + qualitative examples
  - Demo: Gradio link and screenshots
  - Analysis: Effect of heads/layers, attention visualization, response length control
  - Conclusion: Results, limitations, future work

## License
Educational use only. Check dataset license on Kaggle.
