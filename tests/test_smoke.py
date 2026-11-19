from app.storage import load_corpus


def test_corpus_loads():
    rows = load_corpus()
    assert len(rows) > 50
    assert {"blacksmith", "guard", "wizard"} <= {n for n, _ in rows}


from app.cli import main


def test_hello(capsys):
    main(["hello"])
    assert "ready" in capsys.readouterr().out


def test_generate_multiple_and_save(tmp_path, capsys):
    out = tmp_path / "lines.txt"
    main(["generate", "welcome", "--npc", "innkeeper", "--count", "3", "--seed", "1", "--save", str(out)])
    assert capsys.readouterr().out.count("INNKEEPER:") == 3
    assert len(out.read_text().splitlines()) == 3


def test_predict(capsys):
    main(["predict", "halt", "--npc", "guard"])
    assert "!" in capsys.readouterr().out


def test_polish_without_provider_returns_line(monkeypatch):
    from app.llm import polish
    monkeypatch.setenv("LLM_PROVIDER", "")
    assert polish("Halt!", "guard") == "Halt!"
