import os
import json
import requests

_SYSTEM_PROMPT = open("query_router.py").read().split('_SYSTEM_PROMPT = """')[1].split('"""')[0]

QUESTIONS = [
    "How much is Victor Wembanyama worth in a trade?",
    "Which player has the best defensive instincts based on the numbers?",
    "What's Clingan's points per game?",
]

key = os.environ["GROQ_API_KEY"]

for q in QUESTIONS:
    print(f"\nQ: {q}")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
        },
        timeout=30,
    )
    if r.status_code == 200:
        raw = r.json()["choices"][0]["message"]["content"]
        print(f"  HTTP: 200 OK")
        print(f"  Raw response: {repr(raw)}")
        try:
            parsed = json.loads(raw.strip().strip("```json").strip("```").strip())
            print(f"  Parsed JSON: {parsed}")
        except Exception as e:
            print(f"  JSON parse error: {e}")
    else:
        print(f"  HTTP: {r.status_code}")
        print(f"  Error: {r.text[:300]}")
