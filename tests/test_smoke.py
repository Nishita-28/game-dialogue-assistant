from app.storage import load_corpus


def test_corpus_loads():
    rows = load_corpus()
    assert len(rows) > 50
    assert {"blacksmith", "guard", "wizard"} <= {n for n, _ in rows}
