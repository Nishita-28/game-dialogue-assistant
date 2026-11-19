"""Preprocessing and n-gram model."""
import re

BOS, EOS = "<s>", "</s>"
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?]")


def tokenize(text):
    """Lowercase, keep words and basic punctuation."""
    return _TOKEN.findall(text.lower())


def split_sentences(tokens):
    """Split tokens at . ! ?"""
    out, cur = [], []
    for t in tokens:
        cur.append(t)
        if t in ".!?":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def preprocess(rows):
    """(npc, text) rows -> {npc: [sentence tokens]}."""
    data = {}
    for npc, text in rows:
        data.setdefault(npc, []).extend(split_sentences(tokenize(text)))
    return data


def detokenize(tokens):
    """Join tokens back into a readable line."""
    text = ""
    for t in tokens:
        text += t if t in ".,!?" or not text else " " + t
    return re.sub(r"(^|[.!?] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
