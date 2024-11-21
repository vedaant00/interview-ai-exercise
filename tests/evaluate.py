"""Evaluation Script for Retrieval System with Similarity Metrics."""

import requests
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import nltk
nltk.download('punkt')

# ---------------- Query and Ground Truth Setup ----------------

# Define the queries and their expected ground truth
queries_with_ground_truth = [
    {
        "query": "How do you authenticate to the StackOne API?",
        "expected_context": ["Details about the `/connect_sessions/authenticate` endpoint."]
    },
    {
        "query": "What is the default expiry of the session token?",
        "expected_context": ["The default session token expiry is 24 hours."]
    },
    {
        "query": "What fields must be sent to create a course on an LMS?",
        "expected_context": ["Required fields: course_name, description, duration."]
    },
    {
        "query": "Can I retrieve all linked accounts with Workday provider?",
        "expected_context": ["Yes, linked accounts can be retrieved using the Workday integration."]
    },
    {
        "query": "What is the response body when listing an employee?",
        "expected_context": ["Response includes employee_id, name, and position."]
    },
    {
        "query": "What is the weather today?",
        "expected_context": []
    },
    {
        "query": "Tell me a joke.",
        "expected_context": []
    },
    {
        "query": "What is the capital of France?",
        "expected_context": []
    },
    {
        "query": "What is the meaning of life?",
        "expected_context": []
    },
]

# Define the API URL
api_url = "http://localhost:80/chat"

# ---------------- Similarity Metric Functions ----------------

def bm25_score(query: str, retrieved_context: list[str]) -> float:
    """
    Calculate BM25 score for the query and retrieved context.

    Args:
        query (str): The user query.
        retrieved_context (list[str]): The list of retrieved context strings.

    Returns:
        float: The highest BM25 score across the retrieved context.
    """
    if not retrieved_context:
        return 0.0

    tokenized_corpus = [word_tokenize(doc.lower()) for doc in retrieved_context]
    tokenized_query = word_tokenize(query.lower())
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    return max(scores) if scores else 0.0

def tfidf_cosine_similarity(query: str, retrieved_context: list[str]) -> float:
    """
    Calculate TF-IDF Cosine Similarity for the query and retrieved context.

    Args:
        query (str): The user query.
        retrieved_context (list[str]): The list of retrieved context strings.

    Returns:
        float: The highest cosine similarity score across the retrieved context.
    """
    if not retrieved_context:
        return 0.0

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query] + retrieved_context)
    cosine_scores = cosine_similarity(vectors[0], vectors[1:])
    return max(cosine_scores[0]) if cosine_scores.size > 0 else 0.0

def jaccard_similarity(query: str, retrieved_context: list[str]) -> float:
    """
    Calculate Jaccard Similarity for the query and retrieved context using bigrams.

    Args:
        query (str): The user query.
        retrieved_context (list[str]): The list of retrieved context strings.

    Returns:
        float: The highest Jaccard similarity score across the retrieved context.
    """
    query_tokens = set(ngrams(word_tokenize(query.lower()), 2))
    best_score = 0.0
    for context in retrieved_context:
        context_tokens = set(ngrams(word_tokenize(context.lower()), 2))
        intersection = query_tokens.intersection(context_tokens)
        union = query_tokens.union(context_tokens)
        score = len(intersection) / len(union) if union else 0.0
        best_score = max(best_score, score)
    return best_score

# ---------------- Evaluation Function ----------------

def evaluate():
    """
    Evaluate the retrieval system using similarity metrics (BM25, TF-IDF, Jaccard).

    Saves results to a CSV file for further analysis.
    """
    results = []

    for item in queries_with_ground_truth:
        query = item["query"]
        print(f"Evaluating Query: {query}")

        # Send the query to the API
        try:
            response = requests.post(api_url, json={"query": query})
            print("Raw API Response:", response.text)  # Debug raw API response
            if response.status_code == 200:
                response_data = response.json()
                actual_response = response_data.get("message", "No message in response")
                retrieved_context = response_data.get("context", "")  # Extract context as a string
                print("Retrieved Context (Frontend Debug):", retrieved_context)  # Debug retrieved chunks
            else:
                actual_response = f"Error: Received status code {response.status_code}"
                retrieved_context = ""
        except Exception as e:
            actual_response = f"Error: {e}"
            retrieved_context = ""

        # Compute similarity scores
        bm25 = bm25_score(query, [retrieved_context])
        tfidf_cosine = tfidf_cosine_similarity(query, [retrieved_context])
        jaccard = jaccard_similarity(query, [retrieved_context])

        # Append results
        results.append({
            "query": query,
            "response": actual_response,
            "retrieved_context": retrieved_context,
            "bm25_score": round(bm25, 2),
            "tfidf_cosine_similarity": round(tfidf_cosine, 2),
            "jaccard_similarity": round(jaccard, 2),
        })

    # Save results to CSV
    with open("./tests/evaluation-metrics/evaluation_results_with_similarity.csv", mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "query", "response", "retrieved_context", "bm25_score", "tfidf_cosine_similarity", "jaccard_similarity"
        ])
        writer.writeheader()
        writer.writerows(results)

    print("Evaluation complete. Results saved to evaluation_results_with_similarity.csv.")

# ---------------- Entry Point ----------------

if __name__ == "__main__":
    evaluate()