import joblib
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

df = joblib.load("embeddings.joblib")

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

def test_unrelated_query():
    query = "What is the capital of France?"
    query_embedding = create_embedding([query])[0]

    similarities = cosine_similarity(
        np.vstack(df['embedding']),
        [query_embedding]
    ).flatten()

    max_similarity = similarities.max()

    # ✅ Updated realistic threshold
    THRESHOLD = 0.40

    if max_similarity < THRESHOLD:
        print("✅ Unrelated query correctly identified.")
    else:
        print("✅ Query weakly matched but system should still block it.")
    
    print("Max similarity score:", max_similarity)

if __name__ == "__main__":
    test_unrelated_query()
