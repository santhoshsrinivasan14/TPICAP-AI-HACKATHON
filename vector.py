import os
import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(
        self,
        csv_path: str = "courses.csv",
        index_path: str = "course_index.faiss",
        meta_path: str  = "course_metadata.pkl",
        embed_model: str = "all-MiniLM-L6-v2",
    ):
        self.csv_path   = csv_path
        self.index_path = index_path
        self.meta_path  = meta_path
        self.embed_model= embed_model

        df = pd.read_csv(self.csv_path)
        df.columns = [c.strip() for c in df.columns]  # <-- Clean headers

        self.titles      = df["title"].astype(str).tolist()
        self.descriptions= df["description"].astype(str).tolist()

        self.embedder = SentenceTransformer(self.embed_model)

        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            texts = [
                (t if pd.notna(t) else "") + " – " + (d if pd.notna(d) else "")
                for t, d in zip(self.titles, self.descriptions)
            ]
            embs = self.embedder.encode(texts, show_progress_bar=True)
            embs = np.array(embs, dtype="float32")

            dim = embs.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embs)
            faiss.write_index(index, self.index_path)

            with open(self.meta_path, "wb") as f:
                pickle.dump({
                    "titles":      self.titles,
                    "descriptions": self.descriptions
                }, f)

        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        self.titles       = meta["titles"]
        self.descriptions = meta["descriptions"]

    def invoke(self, question: str, top_k: int = 5) -> str:
        q_emb = self.embedder.encode([question]).astype("float32")
        _, idxs = self.index.search(q_emb, top_k)
        idxs = idxs[0].tolist()

        reviews = []
        for i in idxs:
            reviews.append(f"**{self.titles[i]}**: {self.descriptions[i]}")
        return "\n\n".join(reviews)
