"""Corpus loading and saving output."""
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "dialogues.txt"


def load_corpus(path=DEFAULT_CORPUS):
    """Read 'NPC: line' rows into (npc, text) pairs."""
    rows = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or ":" not in raw:
            continue
        npc, text = raw.split(":", 1)
        rows.append((npc.strip().lower(), text.strip()))
    return rows


def save_dialogues(lines, path, npc="any"):
    """Append generated lines to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{npc.upper()}: {line}\n")
    return path
