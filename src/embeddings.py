import math
import os
from dotenv import load_dotenv
from pathlib import Path
import voyageai

load_dotenv(Path(__file__).parent.parent / ".env")

client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

def get_embedding(text: str) -> list[float]:
    result = client.embed([text], model="voyage-3")
    return result.embeddings[0]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    result = client.embed(texts, model="voyage-3")
    return result.embeddings


# ── Similitud coseno ──────────────────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ── Base de conocimiento en memoria ──────────────────────────────────────────

class InMemoryVectorStore:
    def __init__(self):
        self.documents: list[str] = []
        self.embeddings: list[list[float]] = []

    def add(self, documents: list[str]) -> None:
        new_embeddings = get_embeddings(documents)
        self.documents.extend(documents)
        self.embeddings.extend(new_embeddings)
        print(f"Added {len(documents)} documents. Total: {len(self.documents)}")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_embedding = get_embedding(query)

        scored = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            scored.append({"document": self.documents[i], "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def search_with_threshold(self, query: str, min_score: float) -> list[dict]:
        query_embedding = get_embedding(query)
        scored = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            scored.append({"document": self.documents[i], "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return [result for result in scored if result["score"] > min_score]


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Chunks de un contrato financiero
    contract_chunks = [
        "The loan amount is $50,000 USD with an interest rate of 12.5% per annum.",
        "Repayment shall be made in 36 monthly installments of $1,670 USD.",
        "Late payments incur a penalty of 2% per month on the outstanding balance.",
        "The borrower's property at 123 Main Street is pledged as collateral.",
        "Default occurs after 3 consecutive missed payments.",
        "Early repayment before 12 months incurs a fee of 3% of remaining balance.",
        "The lender is ABC Bank S.A. and the borrower is John Smith.",
        "The agreement is governed by the laws of Colombia.",
    ]

    store = InMemoryVectorStore()
    store.add(contract_chunks)

    queries = [
        "What happens if I miss payments?",
        "How much is the monthly payment?",
        "What is the collateral for this loan?",
        "Who are the parties in this contract?",
    ]

    print("\n" + "="*60)
    print("Search with threshold (min_score=0.45):")
    for query in queries:
        print(f"\nQuery: {query}")
        results = store.search_with_threshold(query, min_score=0.45)
        if results:
            for r in results:
                print(f"  [{r['score']:.3f}] {r['document']}")
        else:
            print("  No results above threshold")