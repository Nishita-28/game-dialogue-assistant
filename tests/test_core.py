from app.core import detokenize, preprocess, tokenize


def test_tokenize():
    assert tokenize("Halt! Who goes there?") == ["halt", "!", "who", "goes", "there", "?"]


def test_preprocess_splits_by_npc_and_sentence():
    data = preprocess([("guard", "Halt! Move along.")])
    assert data == {"guard": [["halt", "!"], ["move", "along", "."]]}


def test_detokenize():
    assert detokenize(["halt", "!", "who", "goes", "there", "?"]) == "Halt! Who goes there?"


from app.core import NGramModel

SENTS = [["the", "fire", "is", "hot", "."], ["the", "steel", "is", "ready", "."]]


def test_probs_sum_to_one():
    m = NGramModel(n=2, k=0.5).fit(SENTS)
    assert abs(sum(m.prob(w, ["the"]) for w in m.vocab) - 1) < 1e-9


def test_smoothing_gives_unseen_words_mass():
    m = NGramModel(n=3).fit(SENTS)
    assert m.prob("ready", ["the", "fire"]) > 0
    assert m.prob("is", ["the", "fire"]) > m.prob("ready", ["the", "fire"])


def test_backoff_on_unknown_context():
    m = NGramModel(n=3).fit(SENTS)
    seen, order = m.candidates(["dragon", "is"])
    assert order == 1 and set(seen) == {"hot", "ready"}
