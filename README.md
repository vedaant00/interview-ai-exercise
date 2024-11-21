Here is an updated **README** file incorporating everything you've implemented for the project:

---

# Retrieval-Augmented Generation (RAG) System

> A simple yet powerful RAG example, enhanced for maximum retrieval accuracy of an API Assistant.

---

## **Overview**

This project demonstrates a Retrieval-Augmented Generation (RAG) system that fetches contextually relevant information from a set of OpenAPI specifications and generates responses using OpenAI's language models. The system is designed for high accuracy, user-friendly interaction, and flexibility in extending retrieval methods.

---

## **Project Requirements**

### **Python Environment**

- Install [Pyenv](https://github.com/pyenv/pyenv) to manage Python versions and virtual environments:
  ```bash
  curl -sSL https://pyenv.run | bash
  ```
  - Add these lines to your `~/.bashrc` or `~/.zshrc` for pyenv functionality:
    ```bash
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    eval "$(pyenv init --path)"
    ```
  - Restart your shell and install Python:
    ```bash
    pyenv install 3.11.4
    ```

### **Dependency Management**

- Install [Poetry](https://python-poetry.org) for dependency management:
  ```bash
  curl -sSL https://install.python-poetry.org | python - --version 1.5.1
  ```
  - Ensure Python 3.11.4 is set globally before installing Poetry:
    ```bash
    pyenv global 3.11.4
    ```

### **Docker**

- Install [Docker Engine](https://docs.docker.com/engine/install/) to containerize and run the API.

---

## **Installation**

### **1. a. Create Virtual Environment**

Set up the virtual environment for the project:

```bash
pyenv virtualenv 3.11.4 rag-system
pyenv local rag-system
```

### **1. b. Create Virtual Environment Using `venv`**

If you're not using `pyenv`, you can create a virtual environment using Python's built-in `venv` module:

```bash
# Create the virtual environment
python3 -m venv rag-system

# Activate the virtual environment
# On Linux/macOS
source rag-system/bin/activate

# On Windows
rag-system\Scripts\activate
```

Once activated, you can proceed to install the project dependencies. To deactivate the virtual environment later, simply run:

```bash
deactivate
```

### **2. Install Dependencies**

Use Poetry to install Python dependencies:

```bash
poetry install --no-root
```

### **3. Install Git Hooks**

Install git hooks for pre-commit checks:

```bash
poetry run pre-commit install
```

---

## **Environment Variables**

- Copy `.env_example` to `.env` and fill in the required values (e.g., OpenAI API key, embedding model, etc.).

---

## **API Details**

### **Backend**
The project includes an API built with [FastAPI](https://fastapi.tiangolo.com/), containerized using Docker.

1. **Start the API**:
   - Using Docker:
     ```bash
     make start-api
     ```
   - Without Docker:
     ```bash
     make dev-api
     ```

2. **API Endpoints**:
   - `/health`: Check API health.
   - `/load`: Load OpenAPI specification data into the vector store and precompute embeddings.
   - `/chat`: Handle user queries and return responses based on relevant context.

3. **Swagger Documentation**:
   - Access Swagger UI at `/docs`.

---

## **Frontend**

The frontend is built with [Streamlit](https://streamlit.io/) for interactive querying.

1. Start the frontend:
   ```bash
   make start-app
   ```

2. Use the web interface to test API queries interactively.

---

## **Key Features**

1. **OpenAPI Specification Support**:
   - Retrieves context from 7 OpenAPI specs as mentioned in Notion.

2. **Retrieval System**:
   - Uses OpenAI embeddings to match queries with context from vectorized OpenAPI specs.
   - Filters irrelevant queries and responds appropriately.

3. **Evaluation Metrics**:
   - **BM25**: Measures term relevance using ranking functions.
   - **TF-IDF Cosine Similarity**: Measures similarity based on term frequency-inverse document frequency.
   - **Jaccard Similarity**: Measures overlap of bigrams (2-word sequences).

4. **Embeddings Management**:
   - Efficiently generates and stores embeddings for large datasets using batching and Pickle.

5. **Graceful Limitations**:
   - Returns a user-friendly message for out-of-scope or irrelevant queries.

---

## **Testing**

1. **Unit Tests**:
   Run tests with:
   ```bash
   pytest tests --cov src
   ```
   or:
   ```bash
   make test
   ```

2. **Evaluate Retrieval System**:
   Evaluate system performance with similarity metrics:
   ```bash
   python tests/evaluate.py
   ```
   Results are saved in `evaluation_results_with_similarity.csv`.

---

## **Improvements Made**

1. **Added Contextual Similarity Metrics**:
   - Implemented BM25, TF-IDF, and Jaccard similarity metrics for evaluation to better assess the quality of retrieval.

2. **Embeddings Optimization**:
   - Improved embedding generation with batching and streamlined storage using Pickle for efficient processing of large datasets.
   - Leveraged OpenAI's **Ada (text-embedding-ada-002)** model for high-quality, lightweight embeddings, ensuring efficient and accurate similarity calculations.

3. **Relevance Filtering**:
   - Introduced a query relevance filter using cosine similarity on embeddings, ensuring only relevant context is retrieved.

4. **Improved Error Handling**:
   - Handles empty or malformed responses gracefully, providing user-friendly messages for irrelevant or out-of-scope queries.

5. **Expanded Functionality**  
   - **Supports Filtering Irrelevant Queries**  
     - The system filters out irrelevant queries based on context similarity thresholds, enhancing system responsiveness and improving the overall user experience.  
     - **Current Threshold Setting**  
       - The similarity threshold is currently set low to accommodate the limitations of the embedding model being used.  
     - **Reason**  
       - The embeddings are generated using a generalized model (`text-embedding-ada-002` or `text-embedding-3-small`), which may not capture domain-specific nuances with high precision. A low threshold ensures that even moderately relevant queries are processed without being prematurely discarded.  

---

## **Stretch Goals**

- **Dynamic Threshold Adjustment**: Implement dynamic adjustment of relevance thresholds based on query types or user preferences to improve accuracy.
- **Advanced Retrieval Techniques**: Explore hybrid search techniques combining BM25 and embedding-based retrieval for better relevance scoring.
- **Query Caching**: Implement caching mechanisms for frequent queries to reduce response time and improve system efficiency.
- **Improved Embedding Models**: Experiment with other state-of-the-art embedding models or fine-tuned models for domain-specific tasks.
- **Interactive Evaluation Dashboard**: Build a dashboard to visualize evaluation metrics (BM25, TF-IDF, Jaccard) and track system performance over time.
- **Incremental Indexing**: Add support for incremental updates to the vector store without requiring a full reloading of embeddings.

---

## **How to Get Started**

1. Review `src/constants.py` to understand the configuration.
2. Explore the API in `src/main.py` for detailed logic.
3. Test the system using provided evaluation scripts.