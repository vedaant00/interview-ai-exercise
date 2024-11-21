"""Retrieve Relevant Chunks from a Vector Store."""

from src.llm.embeddings import openai_ef


def get_relevant_chunks(collection, query, k=5, similarity_threshold=0.0):
    """
    Retrieve relevant chunks from a vector store based on query similarity.

    Args:
        collection (chromadb.Collection): The vector store collection to query.
        query (str): The user query to find relevant chunks for.
        k (int, optional): Number of top results to retrieve. Defaults to 5.
        similarity_threshold (float, optional): Minimum similarity score to include a result. Defaults to 0.0.

    Returns:
        list[str]: A list of relevant chunks (documents) that meet the similarity threshold.
    """
    # Query the vector store collection
    results = collection.query(query_texts=[query], n_results=k)

    # Extract documents and scores from the results
    documents = results.get("documents", [[]])[0]  # Retrieve top documents
    scores = results.get("scores", [0])            # Retrieve corresponding similarity scores

    # Filter documents based on the similarity threshold
    filtered_chunks = [
        doc for doc, score in zip(documents, scores) if score >= similarity_threshold
    ]

    return filtered_chunks