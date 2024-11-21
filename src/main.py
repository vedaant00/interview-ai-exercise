"""FastAPI Application for RAG System with OpenAI Integration."""

from fastapi import FastAPI
import numpy as np
import pickle
import json
from src.constants import SETTINGS, chroma_client, openai_client
from src.llm.completions import create_prompt, get_completion
from src.llm.embeddings import openai_ef
from src.loading.document_loader import (
    add_documents,
    build_docs,
    get_json_data,
    split_docs,
)
from src.models import ChatOutput, ChatQuery, HealthRouteOutput, LoadDocumentsOutput
from src.retrieval.retrieval import get_relevant_chunks
from src.retrieval.vector_store import create_collection

# FastAPI instance
app = FastAPI()

# Create a collection for the vector store
collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)


# ---------------- Helper Functions ----------------


def embed_text_with_openai(text: str) -> np.ndarray:
    """
    Generate an embedding for text using OpenAIEmbeddingFunction.

    Args:
        text (str): Input text to embed.

    Returns:
        np.ndarray: A 1D array representing the embedding.
    """
    embedding = openai_ef([text])  # Pass text as a list to support batching
    if isinstance(embedding, list) and len(embedding) == 1:
        return np.array(embedding[0])  # Return the first embedding
    raise ValueError(f"Unexpected embedding format: {embedding}")


def cosine_similarity_openai(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    vec1 = np.array(vec1).flatten()  # Ensure vectors are 1D
    vec2 = np.array(vec2).flatten()
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def is_query_relevant_openai(query: str, doc_embeddings: list[np.ndarray], threshold: float = 0) -> tuple[bool, float]:
    """
    Check if a query is relevant based on similarity with document embeddings.

    Args:
        query (str): The input query.
        doc_embeddings (list[np.ndarray]): List of document embeddings.
        threshold (float): Similarity threshold to consider relevance.

    Returns:
        tuple[bool, float]: Whether the query is relevant and the maximum similarity score.
    """
    # Ensure all embeddings are consistent in shape
    if not all(len(doc_embedding) == len(doc_embeddings[0]) for doc_embedding in doc_embeddings):
        raise ValueError("Inconsistent document embedding shapes detected.")

    # Generate query embedding
    query_embedding = embed_text_with_openai(query)
    if len(query_embedding) != len(doc_embeddings[0]):
        raise ValueError(
            f"Shape mismatch: query embedding {len(query_embedding)} vs document embedding {len(doc_embeddings[0])}"
        )

    # Compute similarities
    similarities = [
        cosine_similarity_openai(query_embedding, doc_embedding)
        for doc_embedding in doc_embeddings
    ]
    max_similarity = max(similarities)
    return max_similarity >= threshold, max_similarity


def save_embeddings_pickle(embeddings: list[np.ndarray], filename: str = "embeddings.pkl"):
    """
    Save embeddings to a file using Pickle.

    Args:
        embeddings (list[np.ndarray]): List of embeddings to save.
        filename (str): Filepath for saving embeddings.
    """
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings_pickle(filename: str = "embeddings.pkl") -> list[np.ndarray]:
    """
    Load embeddings from a Pickle file.

    Args:
        filename (str): Filepath to load embeddings from.

    Returns:
        list[np.ndarray]: List of embeddings.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def generate_embeddings_in_batches(documents: list, batch_size: int = 10) -> list[np.ndarray]:
    """
    Generate embeddings for documents in batches using OpenAIEmbeddingFunction.

    Args:
        documents (list): List of documents to embed.
        batch_size (int): Number of documents per batch.

    Returns:
        list[np.ndarray]: List of embeddings for the documents.
    """
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch]
        batch_embeddings = openai_ef(batch_texts)  # Generate embeddings for the batch
        embeddings.extend(batch_embeddings)
    return embeddings


# ---------------- API Routes ----------------

@app.get("/health")
def health_check_route() -> HealthRouteOutput:
    """
    Health check route to verify API is running.

    Returns:
        HealthRouteOutput: Status of the API.
    """
    return HealthRouteOutput(status="ok")

@app.get("/load")
async def load_docs_route() -> LoadDocumentsOutput:
    """
    Load documents into the vector store and precompute embeddings.

    Returns:
        LoadDocumentsOutput: Status of the load operation.
    """
    global document_embeddings

    # Fetch and process document data
    json_data_list = get_json_data()
    all_documents = []

    for json_data in json_data_list:
        documents = build_docs(json_data)
        documents = split_docs(documents)
        all_documents.extend(documents)

    add_documents(collection, all_documents)

    # Generate and save embeddings
    document_embeddings = generate_embeddings_in_batches(all_documents, batch_size=10)
    save_embeddings_pickle(document_embeddings)

    print(f"Number of documents in collection: {collection.count()}")
    return LoadDocumentsOutput(status="ok")

@app.post("/chat")
def chat_route(chat_query: ChatQuery) -> dict:
    """
    Process a chat query and return a response based on document context.

    Args:
        chat_query (ChatQuery): Input query.

    Returns:
        dict: Chat response with serialized context and message.
    """
    try:
        # Load precomputed embeddings
        document_embeddings = load_embeddings_pickle()
    except FileNotFoundError:
        return {
            "message": "Document embeddings are not loaded. Please run the /load endpoint first.",
            "context": "",
        }

    # Check query relevance
    is_relevant, max_similarity = is_query_relevant_openai(chat_query.query, document_embeddings)
    print(f"is_relevant: {is_relevant}, max_similarity: {max_similarity}")
    if not is_relevant:
        return {
            "message": "This query does not seem to be related to the API documentation. Please refine your question.",
            "context": "",
            "similarity": round(max_similarity, 2),
        }

    # Retrieve relevant chunks
    relevant_chunks = get_relevant_chunks(
        collection=collection, query=chat_query.query, k=SETTINGS.k_neighbors
    )
    print("Relevant Chunks (Debug):", relevant_chunks)

    if not relevant_chunks:
        return {
            "message": "I'm sorry, I couldn't find relevant information for your query.",
            "context": "",
        }

    # Serialize relevant chunks into a single string
    serialized_chunks = "\n".join(relevant_chunks)  # Join chunks with newline for readability

    # Generate prompt and get response
    prompt = create_prompt(query=chat_query.query, context=relevant_chunks)
    result = get_completion(client=openai_client, prompt=prompt, model=SETTINGS.openai_model)

    # Debug: Ensure serialized_chunks is being included
    print("Serialized Context for Response:", serialized_chunks)

    # Return the result and the serialized chunks as context
    return {
        "message": result,
        "context": serialized_chunks,  # Send serialized chunks as a single string
    }


# ---------------- Entry Point ----------------

if __name__ == "__main__":
    import uvicorn

    # Start the FastAPI application
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)