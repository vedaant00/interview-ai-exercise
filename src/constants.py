"""Constants and Configuration Setup for the Project."""

import chromadb
from openai import OpenAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings


# ---------------- Settings Class ----------------


class Settings(BaseSettings):
    """
    Configuration for the application.

    This class uses Pydantic's BaseSettings to read configuration values
    from environment variables or a `.env` file. It securely handles sensitive
    data such as API keys.

    Attributes:
        openai_api_key (SecretStr): Secure API key for OpenAI.
        openai_model (str): Default OpenAI model to be used (e.g., 'gpt-4o').
        embeddings_model (str): Model used for generating embeddings.
        collection_name (str): Name of the ChromaDB collection.
        chunk_size (int): Size of document chunks for processing.
        k_neighbors (int): Number of neighbors for vector similarity queries.
        docs_urls (list[str]): List of OpenAPI specification URLs for processing.

    Notes:
        - The `SecretStr` type ensures sensitive data like `openai_api_key`
          is not exposed in logs or traces.
        - Environment variables are loaded automatically from a `.env` file.
    """

    class Config:
        """
        Pydantic Settings Configuration.

        - Specifies the path to the `.env` file for loading environment variables.
        """
        env_file = ".env"

    # Secure environment variables
    openai_api_key: SecretStr
    openai_model: str = "gpt-4o"
    embeddings_model: str = "text-embedding-ada-002" # text-embedding-3-small

    # Application constants
    collection_name: str = "documents"
    chunk_size: int = 1000
    k_neighbors: int = 5

    # List of OpenAPI specification URLs for documentation processing
    docs_urls: list[str] = [
        "https://api.eu1.stackone.com/oas/stackone.json",
        "https://api.eu1.stackone.com/oas/hris.json",
        "https://api.eu1.stackone.com/oas/ats.json",
        "https://api.eu1.stackone.com/oas/lms.json",
        "https://api.eu1.stackone.com/oas/iam.json",
        "https://api.eu1.stackone.com/oas/crm.json",
        "https://api.eu1.stackone.com/oas/marketing.json",
    ]


# ---------------- Settings Initialization ----------------

# Initialize the SETTINGS object for use throughout the project
SETTINGS = Settings()  # type: ignore


# ---------------- Client Instances ----------------


# OpenAI Client
openai_client = OpenAI(api_key=SETTINGS.openai_api_key.get_secret_value())
"""
The OpenAI client is configured using the API key defined in `SETTINGS`.
"""

# ChromaDB Persistent Client
chroma_client = chromadb.PersistentClient(path="./.chroma_db")
"""
ChromaDB client instance for managing the persistent database.
- The database is stored at the relative path `./.chroma_db`.
"""