from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
import speech_recognition as sr

class EmbeddingIndex:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.texts = []
        self.vecs = None  # shape: (N, d)

    def add_texts(self, texts):
        emb = self.model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
        if self.vecs is None:
            self.vecs = emb
        else:
            self.vecs = np.vstack([self.vecs, emb])
        self.texts.extend(texts)

    def save(self, dirpath="emb_index"):
        p = Path(dirpath)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "vectors.npy", self.vecs)
        with open(p / "texts.json", "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dirpath="emb_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        idx = cls(model_name)
        p = Path(dirpath)
        idx.vecs = np.load(p / "vectors.npy")
        with open(p / "texts.json", "r", encoding="utf-8") as f:
            idx.texts = json.load(f)
        return idx

    def search(self, query, top_k=1):
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self.vecs @ q
        top_idx = np.argpartition(-sims, range(top_k))[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(self.texts[i], float(sims[i])) for i in top_idx]

# Define your base commands (flexible wording here)
docs = [
    "move left", "rotate left", "turn left",
    "move right", "rotate right", "turn right",
    "move up", "lift up", "go up",
    "move down", "lower down", "go down",
    "open claw", "release", "unclamp",
    "close claw", "grab", "clamp"
]

# Build the index once
idx = EmbeddingIndex()
idx.add_texts(docs)
idx.save()

# Function to map recognized text to base actions
def interpret_commands(text):
    steps = []
    parts = [p.strip() for p in text.replace("then", ",").replace("and", ",").split(",") if p.strip()]
    for part in parts:
        match, score = idx.search(part, top_k=1)[0]
        steps.append(match)
    return steps

# Function to simulate sending commands to Arduino
def execute_steps(steps):
    for step in steps:
        print(f"Executing: {step}")
        # Here is where you'd send a serial command to Arduino like:
        # arduino.write(step_code(step))
        # Example:
        # if step in ["move left", "rotate left", "turn left"]:
        #     arduino.write(b'L')
        # elif step in ["close claw", "grab", "clamp"]:
        #     arduino.write(b'C')

# Get speech input
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say your multi-step command...")
    audio = r.listen(source)

try:
    recorded_text = r.recognize_sphinx(audio)
    print(f"You said: {recorded_text}")
    steps = interpret_commands(recorded_text)
    execute_steps(steps)

except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print(f"Sphinx error: {e}")
