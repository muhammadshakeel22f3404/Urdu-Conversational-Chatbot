import re

# Remove diacritics and standardize Alef and Yeh forms for Urdu
DIACRITICS_RANGES = [
    (0x064B, 0x065F),  # Arabic tashkeel
    (0x0610, 0x061A),  # Additional signs
    (0x06D6, 0x06DC),  # Quranic annotation
    (0x06DF, 0x06E8),
    (0x06EA, 0x06ED),
]
DIACRITICS = "".join(chr(c) for a, b in DIACRITICS_RANGES for c in range(a, b + 1))
DIACRITICS_RE = re.compile(f"[{re.escape(DIACRITICS)}]", re.UNICODE)
TATWEEL_RE = re.compile("\u0640", re.UNICODE)

ALEF_FORMS = "أإآٱ"
YEH_FORMS = "يى"
HEH_FORMS = "ۀة"
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EASTERN_ARABIC_INDIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
LATIN_DIGITS = "0123456789"

def normalize_urdu(text: str) -> str:
    if not text:
        return ""
    # Remove tatweel and diacritics
    text = TATWEEL_RE.sub("", text)
    text = DIACRITICS_RE.sub("", text)

    # Standardize Alef to 'ا'
    for ch in ALEF_FORMS:
        text = text.replace(ch, "ا")

    # Standardize Yeh to 'ی'
    for ch in YEH_FORMS:
        text = text.replace(ch, "ی")

    # Teh marbuta and heh goal -> 'ہ'
    for ch in HEH_FORMS:
        text = text.replace(ch, "ہ")

    # Normalize Arabic-Indic digits to Latin (optional)
    for i, ch in enumerate(ARABIC_INDIC_DIGITS):
        text = text.replace(ch, LATIN_DIGITS[i])
    for i, ch in enumerate(EASTERN_ARABIC_INDIC_DIGITS):
        text = text.replace(ch, LATIN_DIGITS[i])

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text