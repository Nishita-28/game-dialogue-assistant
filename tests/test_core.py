from app.core import detokenize, preprocess, tokenize


def test_tokenize():
    assert tokenize("Halt! Who goes there?") == ["halt", "!", "who", "goes", "there", "?"]


def test_preprocess_splits_by_npc_and_sentence():
    data = preprocess([("guard", "Halt! Move along.")])
    assert data == {"guard": [["halt", "!"], ["move", "along", "."]]}


def test_detokenize():
    assert detokenize(["halt", "!", "who", "goes", "there", "?"]) == "Halt! Who goes there?"
