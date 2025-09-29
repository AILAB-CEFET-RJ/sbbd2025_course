# 📘 SBBD 2025 Short Course – Notebooks

This folder contains the teaching notebooks for the course *Introduction to LLM-Based Agents* (SBBD 2025).

## 🛠️ Installation

First, create and activate the conda environment, then install dependencies:

```bash
pip install -r requirements.txt

## 🖥️ Running on CPU vs. GPU

All HuggingFace models in these notebooks are configured to run on the **CPU**:

```python
model_kwargs={"device": "cpu"}
```

### Why CPU?
- Some entry-level GPUs (e.g., **GeForce GT 1030**) do not support the CUDA compute capabilities required by recent PyTorch and Transformers builds.  
- Using CPU ensures the notebooks run **consistently across all machines**, including student laptops without dedicated GPUs.  
- Results are identical between CPU and GPU. The only difference is **speed** (GPU can be faster, when supported).

### Can I use GPU?
Yes — if your GPU is compatible and you have a working CUDA installation, simply remove the explicit `{"device": "cpu"}` option. For example:

```python
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

or for text generation:

```python
generator = pipeline("text-generation", model="gpt2")  # GPU if available
```

## 📂 Notebook Index

- `00_intro.ipynb` — Environment check and first LLM calls  
- `01_ngram_vs_llm.ipynb` — classic n-gram models x LLMs  
- `02_minimal_agent.ipynb` — LLM as brain, agent as body  
- `03_prompting_patterns.ipynb` — Prompting and interaction patterns  
- `04_tool_calling.ipynb` — Structured tool calling with LangChain  
- `05_rag_pipeline.ipynb` — Retrieval-Augmented Generation (RAG) with ChromaDB  
- `06_text_to_sql.ipynb` — Text-to-SQL pipeline with SQLite  
