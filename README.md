# Game Dialogue Assistant

A command-line tool that game designers can use to prototype NPC dialogue. It trains an **n-gram language model** (built from scratch, with add-k smoothing and backoff) on a corpus of game dialogue. Each NPC gets their own model, so generated lines stay in that character's voice.

```
$ python -m app.cli generate "the" --npc wizard --count 2 --sentences 2 --seed 9
1. WIZARD: The old castle hides a power older than the kingdom itself. It is cursed.
2. WIZARD: The stars speak of a hero from a small village. Magic flows through all living things.
```

## Setup

You need Python 3.10+ (CI uses 3.12) and git.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.cli hello
pytest
```

## Usage

| Command | What it does |
|---|---|
| `python -m app.cli npcs` | List the NPCs in the corpus |
| `python -m app.cli generate "PHRASE" --npc guard` | Generate a line that starts with PHRASE |
| `python -m app.cli predict "PHRASE" --npc guard` | Show the most likely next words |
| `python -m app.cli chat --npc innkeeper` | Interactive mode (`:npc <name>` switches NPC, `quit` exits) |

`generate` options:

- `--length 20`: maximum number of words to add
- `--sentences 1`: stop after this many sentences
- `--count 3`: generate several alternative lines
- `--save output/lines.txt`: append the results to a file
- `--seed 42`: make the output repeatable
- `--polish`: optionally tidy the grammar with an LLM (see below)

Global options go before the command: `-n 3` (n-gram order), `-k 0.1` (smoothing), and `--corpus path.txt`.

## How it works

1. **Preprocessing** ([app/core.py](app/core.py)): the text is lowercased, tokenized into words and punctuation, and split into sentences, grouped by NPC.
2. **Model** (`NGramModel`): counts n-grams of every order up to `n`, with `<s>` and `</s>` padding. `prob()` uses add-k smoothing, so unseen words still get some probability. `candidates()` backs off to shorter contexts when a context was never seen.
3. **Generation** (`DialogueModel`): samples the next word from the NPC's own model. It falls back to the shared model only when the NPC has no matching context, which keeps characters consistent. Generation stops at the end of a sentence or at `--length`.
4. **CLI** ([app/cli.py](app/cli.py)): argparse commands for generating, predicting, chatting and saving.

## Corpus

The corpus is [data/dialogues.txt](data/dialogues.txt): original lines for five fantasy NPC archetypes (blacksmith, innkeeper, guard, merchant, wizard), one per line, in the form `NPC: line`. To use your own scripts, add lines in the same format, or point `--corpus` at your own file.

## Optional LLM polish

The core tool needs no API key. To use `--polish`, run `cp .env.example .env` and choose Ollama (local), Groq or Gemini. Never commit `.env`.

## Tests

`pytest` runs tests for tokenizing, smoothing (the probabilities sum to 1), backoff, generation, character consistency and the CLI. GitHub Actions runs them on every push.
