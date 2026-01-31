import hashlib
import numpy as np

class EndeeClient:
    def get_collection(self, name):
        return self

    def search(self, vector, top_k):
        return [
            {"text": "Sample retrieved document chunk 1"},
            {"text": "Sample retrieved document chunk 2"},
        ]


COLLECTION_NAME = "pdf_documents"


def fake_embedding(text: str, dim: int = 384):
    h = hashlib.sha256(text.encode()).digest()
    vec = np.frombuffer(h, dtype=np.uint8)
    return vec[:dim].astype(float).tolist()


def semantic_search(query: str, top_k: int = 3):
    query_embedding = fake_embedding(query)

    client = EndeeClient()
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.search(
        vector=query_embedding,
        top_k=top_k
    )

    return results


if __name__ == "__main__":
    q = input("Enter your question: ")
    results = semantic_search(q)

    print("\nTop retrieved results:\n")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['text']}\n")