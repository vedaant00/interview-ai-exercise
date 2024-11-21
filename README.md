# 🚀 **Retrieval-Augmented Generation (RAG) System**

> 🎯 A simple yet powerful RAG example, enhanced for maximum retrieval accuracy of an API Assistant.

---

## **📌 Overview**

This project demonstrates a Retrieval-Augmented Generation (RAG) system that fetches contextually relevant information from a set of OpenAPI specifications and generates responses using OpenAI's language models. The system is designed for high accuracy, user-friendly interaction, and flexibility in extending retrieval methods.

---

## **🔧 Project Requirements**

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

## **📥 Installation**

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

To deactivate the virtual environment later, simply run:

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

## **🔒 Environment Variables**

- Copy `.env_example` to `.env` and fill in the required values (e.g., OpenAI API key, embedding model, etc.).

---

## **📡 API Details**

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

## **🖥️ Frontend**

The frontend is built with [Streamlit](https://streamlit.io/) for interactive querying.

1. Start the frontend:
   ```bash
   make start-app
   ```

2. Use the web interface to test API queries interactively.

---

## **💡 Key Features**

1. **OpenAPI Specification Support**:
   - Retrieves context from 7 OpenAPI specs as mentioned in the assignment.

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

## **🧪 Testing**

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

## **✨ Improvements Made**

1. **Added Contextual Similarity Metrics**:
   - Implemented BM25, TF-IDF, and Jaccard similarity metrics for evaluation.

2. **Embeddings Optimization**:
   - Improved embedding generation with batching and streamlined storage using Pickle.
   - Leveraged OpenAI's **Ada (text-embedding-ada-002)** model for high-quality embeddings.

3. **Relevance Filtering**:
   - Introduced a query relevance filter using cosine similarity on embeddings.

4. **Improved Error Handling**:
   - Handles empty or malformed responses gracefully.

5. **Expanded Functionality**  
   - Filters irrelevant queries based on context similarity thresholds.  
     - **Current Threshold**: Set low due to limitations in the embedding model (`text-embedding-ada-002`).  
     - **Future Plan**: Fine-tune embeddings to capture domain-specific nuances and dynamically adjust thresholds.  

---

## **🎯 Stretch Goals**

- **Dynamic Threshold Adjustment**: Adjust relevance thresholds dynamically based on query types.
- **Hybrid Retrieval Techniques**: Combine BM25 and embedding-based search for better scoring.
- **Query Caching**: Implement caching for frequently asked queries.
- **Fine-Tuned Embeddings**: Explore domain-specific fine-tuned embeddings.
- **Interactive Evaluation Dashboard**: Visualize metrics like BM25, TF-IDF, and Jaccard.
- **Incremental Indexing**: Support updates to the vector store without full reloads.

---

## **🚀 How to Get Started**

1. Review `src/constants.py` for configuration details.
2. Explore `src/main.py` for API logic.
3. Test and evaluate using the provided scripts.