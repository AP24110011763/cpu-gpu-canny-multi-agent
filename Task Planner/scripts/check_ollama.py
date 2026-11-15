import ollama

reply = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Reply with exactly one word: ready"}],
    options={"num_gpu": 0},   # 0 GPU layers = pure CPU inference
)
print(reply["message"]["content"])
