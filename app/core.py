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


class NGramModel:
    """N-gram model with add-k smoothing and backoff."""

    def __init__(self, n=3, k=0.1):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n, self.k = n, k
        self.counts = {}  # context tuple -> {word: count}
        self.vocab = set()

    def fit(self, sentences):
        for sent in sentences:
            toks = [BOS] * (self.n - 1) + list(sent) + [EOS]
            self.vocab.update(toks[self.n - 1:])
            for i in range(self.n - 1, len(toks)):
                for j in range(self.n):  # all orders
                    ctx = tuple(toks[i - j:i])
                    nxt = self.counts.setdefault(ctx, {})
                    nxt[toks[i]] = nxt.get(toks[i], 0) + 1
        return self

    def _ctx(self, context):
        ctx = [BOS] * (self.n - 1) + list(context)
        return tuple(ctx[len(ctx) - (self.n - 1):]) if self.n > 1 else ()

    def prob(self, word, context=()):
        """Add-k smoothed P(word | last n-1 words)."""
        seen = self.counts.get(self._ctx(context), {})
        total = sum(seen.values())
        return (seen.get(word, 0) + self.k) / (total + self.k * len(self.vocab))

    def candidates(self, context=()):
        """Seen next words, backing off to shorter contexts."""
        ctx = self._ctx(context)
        for j in range(len(ctx), -1, -1):
            seen = self.counts.get(ctx[len(ctx) - j:])
            if seen:
                return seen, j
        return {}, -1
