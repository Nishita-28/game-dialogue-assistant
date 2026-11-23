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


import random

from app.core import DialogueModel
from app.storage import load_corpus


def _model():
    return DialogueModel().fit(preprocess(load_corpus()))


def test_predict_next():
    top = _model().predict_next("guard", "halt", top=1)
    assert top[0][0] == "!"


def test_generate_starts_with_phrase_and_ends_cleanly():
    out = _model().generate("blacksmith", "the fire", rng=random.Random(1))
    assert out.startswith("The fire")
    assert out[-1] in ".!?"


def test_generate_is_reproducible_and_bounded():
    m = _model()
    a = m.generate("wizard", "the stars", max_words=5, rng=random.Random(7))
    b = m.generate("wizard", "the stars", max_words=5, rng=random.Random(7))
    assert a == b and len(tokenize(a)) <= 7


def test_character_consistency():
    m = _model()
    out = m.generate("wizard", "magic", rng=random.Random(3))
    words = set(tokenize(out)) - {".", ",", "!", "?"}
    wizard_vocab = m.npcs["wizard"].vocab
    assert words <= wizard_vocab


def test_detokenize_capitalizes_i():
    assert detokenize(["now", "i", "guard", "this", "gate", "."]) == "Now I guard this gate."
