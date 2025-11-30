import joblib
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Load saved embeddings DataFrame
df = joblib.load("embeddings.joblib")

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

def test_valid_topic_query():
    query = "Where is the box model taught?"
    query_embedding = create_embedding([query])[0]

    similarities = cosine_similarity(
        np.vstack(df['embedding']), 
        [query_embedding]
    ).flatten()

    top_index = similarities.argmax()
    result = df.iloc[top_index]

    assert "box model" in result["text"].lower()
    assert result["start"] >= 0
    assert result["end"] > result["start"]

    print("✅ Test Case 1 Passed: Valid topic retrieval works correctly.")
    print(f"Video: {result['title']} | Time: {result['start']} - {result['end']}")

if __name__ == "__main__":
    test_valid_topic_query()
