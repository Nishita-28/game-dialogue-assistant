"""Optional LLM polish. The n-gram model works without it."""
import json
import os
import urllib.request

PROMPT = "Fix grammar in this {npc} NPC line. Keep its words and tone. Reply with the line only:\n{line}"


def _post(url, body, headers=None):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def polish(line, npc="npc"):
    """Return polished line, or the original if no provider is set."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    p = os.getenv("LLM_PROVIDER", "").lower()
    text = PROMPT.format(npc=npc, line=line)
    try:
        if p == "ollama":
            url = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
            r = _post(url, {"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "prompt": text, "stream": False})
            return r["response"].strip()
        if p == "groq":
            r = _post("https://api.groq.com/openai/v1/chat/completions",
                      {"model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                       "messages": [{"role": "user", "content": text}]},
                      {"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]})
            return r["choices"][0]["message"]["content"].strip()
        if p == "gemini":
            model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
                   f":generateContent?key={os.environ['GEMINI_API_KEY']}")
            r = _post(url, {"contents": [{"parts": [{"text": text}]}]})
            return r["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:  # keep CLI usable offline
        print(f"(polish skipped: {e})")
    return line
