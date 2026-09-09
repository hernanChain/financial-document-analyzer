import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
import voyageai

load_dotenv(Path(__file__).parent.parent / ".env")
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

def get_embeddings(texts: list[str]) -> list[list[float]]:
    result = voyage_client.embed(texts, model="voyage-3")
    return result.embeddings

class VoyageEmbeddingFunction(chromadb.EmbeddingFunction):

    def __init__(self):
        pass

    def __call__(self, input: list[str]) -> list[list[float]]:
        return get_embeddings(input)

class PersistentVectorStore:
    def __init__(self, collection_name: str, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = VoyageEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, documents: list[str], metadata: list[dict] | None = None) -> None:
        existing = self.collection.count()
        ids = [str(existing + i) for i in range(len(documents))]

        if metadata is None:
            metadata = [{}] * len(documents)

        self.collection.add(
            documents = documents,
            metadatas = metadata,
            ids = ids,
        )
        print(f"Added {len(documents)} documents. Total: {self.collection.count()}")

    def search(self, query:str, top_k: int = 3) -> list[dict]:
        results = self.collection.query(
            query_texts = [query],
            n_results = top_k
        )

        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "document": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        return output

    def search_with_filter(self, query: str, filter: dict, top_k: int = 3) -> list[dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where = filter
        )
        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "document": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })
        return output

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection reset.")

if __name__ == "__main__":
    store = PersistentVectorStore(collection_name="contracts")
    #store.reset()

    chunks = [
        ("The loan amount is $50,000 USD with an interest rate of 12.5% per annum.",
         {"contract_id": "loan_001", "section": "terms"}),
        ("Repayment shall be made in 36 monthly installments of $1,670 USD.",
         {"contract_id": "loan_001", "section": "repayment"}),
        ("Late payments incur a penalty of 2% per month on the outstanding balance.",
         {"contract_id": "loan_001", "section": "penalties"}),
        ("The borrower's property at 123 Main Street is pledged as collateral.",
         {"contract_id": "loan_001", "section": "collateral"}),
        ("Default occurs after 3 consecutive missed payments.",
         {"contract_id": "loan_001", "section": "default"}),
        ("The lender is ABC Bank S.A. and the borrower is John Smith.",
         {"contract_id": "loan_001", "section": "parties"}),
        ("The loan amount is $80,000 USD with an interest rate of 9.5% per annum.",
         {"contract_id": "loan_002", "section": "terms"}),
        ("Repayment shall be made in 60 monthly installments of $1,650 USD.",
         {"contract_id": "loan_002", "section": "repayment"}),
        ("The lender is XYZ Finance and the borrower is Maria Garcia.",
         {"contract_id": "loan_002", "section": "parties"}),
    ]

    documents = [c[0] for c in chunks]
    metadata = [c[1] for c in chunks]
    if store.count() == 0:
        store.add(documents, metadata)
    else:
        print(f"Collection already has {store.count()} documents. Skipping add.")

    print("\n=== Normal search ===")
    query = "What is the monthly payment?"
    results = store.search(query, top_k=2)

    for r in results:
        print(f"  [{r['score']:.3f}] [{r['metadata']['contract_id']}] {r['document']}")

    print("\n=== Filtered search (loan_001 only) ===")
    results = store.search_with_filter(
        query="What is the monthly payment?",
        filter={"contract_id": "loan_001"},
        top_k=2,
    )
    for r in results:
        print(f"  [{r['score']:.3f}] [{r['metadata']['contract_id']}] {r['document']}")

    print(f"\nTotal documents in store: {store.count()}")
    print("Run again to verify persistence - documents won't be re-embedded.")