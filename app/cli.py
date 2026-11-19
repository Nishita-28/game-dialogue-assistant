"""Command-line interface."""
import argparse
import random
import sys

from app.core import DialogueModel, preprocess
from app.storage import DEFAULT_CORPUS, load_corpus, save_dialogues


def build(args):
    return DialogueModel(n=args.n, k=args.k).fit(preprocess(load_corpus(args.corpus)))


def check_npc(model, npc):
    if npc not in model.npcs:
        sys.exit(f"Unknown NPC '{npc}'. Try: {', '.join(sorted(model.npcs))}")


def cmd_generate(model, args):
    npc = args.npc.lower()
    check_npc(model, npc)
    rng = random.Random(args.seed)
    lines = [model.generate(npc, args.phrase, args.length, rng, args.sentences)
             for _ in range(args.count)]
    if args.polish:
        from app.llm import polish
        lines = [polish(line, npc) for line in lines]
    for i, line in enumerate(lines, 1):
        print(f"{i}. {npc.upper()}: {line}")
    if args.save:
        print(f"Saved to {save_dialogues(lines, args.save, npc)}")


def cmd_predict(model, args):
    npc = args.npc.lower()
    check_npc(model, npc)
    for word, p in model.predict_next(npc, args.phrase, args.top):
        print(f"{word:<12} {p:.2f}")


def cmd_interactive(model, args):
    npc = args.npc.lower()
    check_npc(model, npc)
    rng = random.Random(args.seed)
    print(f"Talking as {npc.upper()}. Type a phrase, ':npc <name>' to switch, or 'quit'.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("quit", "exit"):
            break
        if text.startswith(":npc "):
            name = text[5:].strip().lower()
            if name in model.npcs:
                npc = name
                print(f"Now {npc.upper()}.")
            else:
                print(f"Unknown. Try: {', '.join(sorted(model.npcs))}")
        elif text:
            print(f"{npc.upper()}: {model.generate(npc, text, args.length, rng, args.sentences)}")


def parser():
    p = argparse.ArgumentParser(prog="python -m app.cli", description="Generate NPC dialogue with an n-gram model.")
    p.add_argument("--corpus", default=DEFAULT_CORPUS, help="dialogue file (NPC: line)")
    p.add_argument("-n", type=int, default=3, help="n-gram order")
    p.add_argument("-k", type=float, default=0.1, help="add-k smoothing")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("hello", help="check setup")
    sub.add_parser("npcs", help="list NPCs")

    g = sub.add_parser("generate", help="generate dialogue")
    g.add_argument("phrase")
    g.add_argument("--npc", default="innkeeper")
    g.add_argument("--length", type=int, default=20, help="max words")
    g.add_argument("--sentences", type=int, default=1)
    g.add_argument("--count", type=int, default=1, help="number of options")
    g.add_argument("--seed", type=int)
    g.add_argument("--save", help="append output to file")
    g.add_argument("--polish", action="store_true", help="tidy with LLM from .env")

    pr = sub.add_parser("predict", help="show likely next words")
    pr.add_argument("phrase")
    pr.add_argument("--npc", default="innkeeper")
    pr.add_argument("--top", type=int, default=5)

    i = sub.add_parser("chat", help="interactive mode")
    i.add_argument("--npc", default="innkeeper")
    i.add_argument("--length", type=int, default=20)
    i.add_argument("--sentences", type=int, default=1)
    i.add_argument("--seed", type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.cmd == "hello":
        print("Hello! Game Dialogue Assistant is ready.")
        return
    model = build(args)
    if args.cmd == "npcs":
        for name, m in sorted(model.npcs.items()):
            print(f"{name:<12} {len(m.vocab)} words")
    elif args.cmd == "generate":
        cmd_generate(model, args)
    elif args.cmd == "predict":
        cmd_predict(model, args)
    elif args.cmd == "chat":
        cmd_interactive(model, args)


if __name__ == "__main__":
    main()
