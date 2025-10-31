import re

# Basic Urdu tokenization:
# - Keeps Urdu words intact
# - Separates punctuation as tokens
# - Works on normalized text
URDU_LETTERS_RANGE = r"\u0600-\u06FF"
PUNCT = r"\.\,\!\?\:\;\-\—\(\)\[\]\{\}\"\'\“\”\‘\’\`،؛؟…"

# regex to capture Urdu words or non-space single characters
TOKEN_RE = re.compile(rf"([{URDU_LETTERS_RANGE}]+|[^\s])", re.UNICODE)

def tokenize(text: str):
    if not text:
        return []
    # Find all tokens: sequences of Urdu letters OR any non-space single char
    tokens = TOKEN_RE.findall(text)
    # Collapse spaces and trivial tokens like multiple dots
    tokens = [t for t in tokens if t.strip() != ""]
    return tokens

def detokenize(tokens):
    # Simple join with spaces, then fix space around punctuation
    s = " ".join(tokens)
    s = re.sub(r"\s+([{}])".format(PUNCT), r"\1", s)
    s = re.sub(r"([(\[{{])\s+".format(), r"\1", s)
    s = re.sub(r"\s+([)\]}}])", r"\1", s)
    # Trim
    return s.strip()