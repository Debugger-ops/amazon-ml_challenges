import re
import unicodedata

LEGAL_MARKERS = {
    "inc", "incorporated", "corp", "corporation", "limited", "ltd",
    "private", "pvt", "llc", "llp", "plc", "company", "co", "dba"
}

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

def strip_accents(text: str) -> str:
    # é -> e, ç -> c, etc. No-op on Devanagari or plain ASCII.
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(ch for ch in nfkd if not unicodedata.combining(ch))

def normalize_name(raw: str) -> dict:
    raw = raw or ""
    has_devanagari = bool(DEVANAGARI_RE.search(raw))

    text = raw.lower()
    text = strip_accents(text)
    text = text.replace('&', ' and ')
    text = re.sub(r"[.,'\"()/\-]", ' ', text)              # punctuation -> space
    text = re.sub(r'[^a-z0-9\u0900-\u097F\s]', ' ', text)  # drop anything stray
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    markers_present = sorted(t for t in tokens if t in LEGAL_MARKERS)
    core_tokens = [t for t in tokens if t not in LEGAL_MARKERS]

    return {
        "normalized_full": text,
        "normalized_core": ' '.join(core_tokens),   # use THIS for similarity scoring
        "tokens": core_tokens,
        "has_devanagari": has_devanagari,
        "legal_markers": markers_present,
    }

def normalize_address(raw: str) -> dict:
    raw = raw or ""
    text = raw.lower()
    text = strip_accents(text)
    text = re.sub(r"[.,#]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = [t for t in text.split(' ') if t]
    numeric_tokens = [t for t in tokens if re.search(r'\d', t)]
    pin_candidates = [t for t in numeric_tokens if re.fullmatch(r'\d{5,6}', t)]

    return {
        "normalized_full": text,
        "tokens": tokens,
        "pin_candidates": pin_candidates,   # likely PIN/ZIP codes, useful as a strong feature
    }