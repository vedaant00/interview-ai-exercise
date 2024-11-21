"""OpenAI Embedding Integration with ChromaDB."""

import chromadb.utils.embedding_functions as embedding_functions
from src.constants import SETTINGS

# ---------------- OpenAI Embedding Initialization ----------------


# Define OpenAIEmbeddingFunction using API key and model from SETTINGS
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=SETTINGS.openai_api_key.get_secret_value(),  # Retrieve the API key securely
    model_name=SETTINGS.embeddings_model                 # Specify the embedding model (e.g., 'text-embedding-ada-002')
)