"""Document Loader for the Retrieval-Augmented Generation (RAG) System."""

import json
from typing import Any

import chromadb
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.constants import SETTINGS
from src.loading.chunk_json import chunk_data
from src.models import Document


# ---------------- Helper Functions ----------------


def get_json_data() -> list[dict[str, Any]]:
    """
    Fetch and parse JSON data from the URLs defined in SETTINGS.

    Returns:
        list[dict[str, Any]]: A list of JSON objects fetched from the specified URLs.

    Raises:
        HTTPError: If any of the HTTP requests fail.
    """
    json_data = []
    for url in SETTINGS.docs_urls:
        response = requests.get(url)
        response.raise_for_status()  # Ensure successful request
        json_data.append(response.json())
    return json_data


def document_json_array(data: list[dict[str, Any]], source: str) -> list[Document]:
    """
    Convert an array of JSON chunks into a list of Document objects.

    Args:
        data (list[dict[str, Any]]): JSON chunks to convert.
        source (str): Metadata source identifier.

    Returns:
        list[Document]: A list of Document objects with metadata.
    """
    return [
        Document(page_content=json.dumps(item), metadata={"source": source})
        for item in data
    ]


def build_docs(data: dict[str, Any]) -> list[Document]:
    """
    Process JSON data into a list of Document objects, chunked by specific attributes.

    Args:
        data (dict[str, Any]): JSON object containing API data.

    Returns:
        list[Document]: A list of Document objects after chunking and conversion.
    """
    docs = []
    for attribute in ["paths", "webhooks", "components"]:
        chunks = chunk_data(data, attribute)  # Extract chunks for the given attribute
        docs.extend(document_json_array(chunks, attribute))
    return docs


def split_docs(docs_array: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks if they are too long.

    Args:
        docs_array (list[Document]): List of original Document objects.

    Returns:
        list[Document]: List of smaller, split Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["}],", "},", "}", "]", " ", ""],  # Define split boundaries
        chunk_size=SETTINGS.chunk_size               # Maximum size for each chunk
    )
    return splitter.split_documents(docs_array)


def add_documents(collection: chromadb.Collection, docs: list[Document]) -> None:
    """
    Add a list of documents to the ChromaDB collection.

    Args:
        collection (chromadb.Collection): The target ChromaDB collection.
        docs (list[Document]): List of Document objects to add.
    """
    collection.add(
        documents=[doc.page_content for doc in docs],  # Content of each document
        metadatas=[doc.metadata or {} for doc in docs],  # Associated metadata
        ids=[f"doc_{i}" for i in range(len(docs))]       # Unique IDs for each document
    )