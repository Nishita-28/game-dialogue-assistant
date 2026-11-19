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


class DialogueModel:
    """One model per NPC plus a shared fallback."""

    def __init__(self, n=3, k=0.1):
        self.n, self.k = n, k
        self.npcs = {}
        self.shared = NGramModel(n, k)

    def fit(self, data):
        for npc, sents in data.items():
            self.npcs[npc] = NGramModel(self.n, self.k).fit(sents)
        self.shared.fit([s for sents in data.values() for s in sents])
        return self

    def _next(self, npc, context):
        # NPC voice wins ties; shared model fills gaps
        model = self.npcs.get(npc)
        own, o1 = model.candidates(context) if model else ({}, -1)
        alt, o2 = self.shared.candidates(context)
        return own if own and o1 >= o2 else alt

    def predict_next(self, npc, phrase, top=3):
        seen = self._next(npc, tokenize(phrase))
        total = sum(seen.values())
        ranked = sorted(seen.items(), key=lambda kv: -kv[1])[:top]
        return [(w, c / total) for w, c in ranked]

    def generate(self, npc, phrase, max_words=20, rng=None, sentences=1):
        import random
        rng = rng or random.Random()
        toks = tokenize(phrase)
        ends = sum(t in ".!?" for t in toks)
        for _ in range(max_words):
            # restart context after each sentence
            last = max((i for i, t in enumerate(toks) if t in ".!?"), default=-1)
            seen = self._next(npc, toks[last + 1:])
            words = list(seen)
            weights = [seen[w] + self.k for w in words]
            w = rng.choices(words, weights)[0]
            if w == EOS:
                break
            toks.append(w)
            if w in ".!?":
                ends += 1
                if ends >= sentences:
                    break
        return detokenize(toks)
