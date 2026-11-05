# Multimodal RAG System for The Batch (DeepLearning.AI)

An end-to-end Multimodal Retrieval-Augmented Generation (RAG) system that indexes and searches news articles and technical illustrations from DeepLearning.AI's newsletter *The Batch*. 

The system enables users to ask natural language questions about recent AI research and industry news, retrieves relevant article passages alongside associated visual figures/diagrams, and synthesizes answers with source citations.

---

## System Architecture

```
User Query
    │
    ├──► [1. Query Embedding] ──► all-MiniLM-L6-v2 (ONNX Runtime)
    │                                     │
    │                                     ▼
    ├──► [2. Multimodal Retrieval] ──► ChromaDB Vector Store
    │         ├── Text Collection: Article chunks (Title + Content)
    │         └── Image Collection: Article figures & visual contexts
    │                                     │
    │                                     ▼
    └──► [3. Context Construction] ──► Top-K passages + Local image paths
                                          │
                                          ▼
         [4. Generative Synthesis] ──► Gemini 2.5 / Ollama / OpenAI / Extractive
                                          │
                                          ▼
         [5. Interactive UI] ────────► Streamlit Interface
                                       - Answer with [1], [2] citations
                                       - Figure gallery & original source links
                                       - Live IR Evaluation Dashboard
```

### Key Modules

1. **Data Ingestion (`src/scraper.py`)**:
   - Parses *The Batch* sitemap and extracts structured articles.
   - Cleans boilerplates (cookie notices, promotional footers).
   - Downloads accompanying figures and illustrations locally into `data/images/`.
   - Stores article metadata and text into `data/the_batch_articles.json`.

2. **Multimodal Database (`src/indexer.py`)**:
   - Implements chunking with sliding context and title prepending.
   - Indexes text chunks and visual context into local persistent ChromaDB collections.
   - Supports text-to-text, text-to-image, and unified multimodal queries.

3. **RAG Engine (`src/rag_engine.py`)**:
   - Orchestrates retrieval, builds grounded prompts, and interfaces with LLM backends:
     - **Google Gemini**: supports native multimodal image grounding (passes retrieved images directly to Gemini 2.5 Flash).
     - **Ollama**: runs completely locally with OSS models (e.g., `qwen2.5-coder:7b`, `gemma2:9b`, `llama3.2`).
     - **OpenAI**: supports `gpt-4o-mini` / `gpt-4o`.
     - **Extractive Fallback**: offline mode synthesizing key passages if no API key or local model daemon is available.

4. **Web UI (`app.py`)**:
   - Interactive Streamlit dashboard with three dedicated tabs:
     - **Q&A & Retrieval**: query input, synthesized response, image carousel, expandable article text with similarity scores.
     - **Visual Media Search**: direct visual search across diagram illustrations.
     - **System Evaluation**: benchmark results, metrics visualization, and live evaluation trigger.

5. **Evaluation Benchmark (`src/evaluate.py`)**:
   - Computes standard Information Retrieval metrics across representative AI domain queries:
     - **Hit@1, Hit@3, Hit@5**
     - **Mean Reciprocal Rank (MRR)**
     - **Multimodal Visual Coverage** (retrieval containing accompanying figures)
     - **Query Latency** (milliseconds)

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 – 3.12
- Linux, macOS, or Windows

### 2. Environment Setup
Clone the repository and create a virtual environment:

```bash
git clone <repo-url>
cd spgai-project

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. API Keys Configuration (Optional)
Copy the example environment file:

```bash
cp .env.example .env
```

If you plan to use Google Gemini for multimodal generation, add your free API key from [Google AI Studio](https://aistudio.google.com/):
```bash
GEMINI_API_KEY="your-gemini-api-key"
```
*(Note: If no API key is provided, the application automatically uses local Ollama or the local extractive synthesizer, ensuring it works out of the box without any external accounts).*

---

## Usage

### 1. Build or Refresh Vector Index
A pre-scraped dataset of 60 recent articles and 100+ images from *The Batch* is included in `data/`. To index it into ChromaDB:

```bash
python -m src.indexer --reset
```

To scrape fresh articles from DeepLearning.AI:
```bash
python -m src.scraper --limit 60
python -m src.indexer --reset
```

### 2. Launch the Streamlit App
Run the interactive user interface:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### 3. Run Benchmark Evaluation
Run the automated retrieval evaluation suite from CLI:

```bash
python -m src.evaluate
```
This generates `data/eval_results.json` and a visualization plot at `data/eval_metrics.png`.

---

## Project Structure

```
spgai-project/
├── app.py                      # Streamlit web user interface
├── requirements.txt            # Python package dependencies
├── .env.example                # Template for environment variables
├── README.md                   # Setup guide and usage instructions
├── REPORT.md                   # Detailed technical report and decisions
├── src/
│   ├── __init__.py
│   ├── scraper.py              # Ingestion pipeline for The Batch
│   ├── indexer.py              # ChromaDB vector store and embeddings
│   ├── rag_engine.py           # Multimodal retrieval and LLM synthesis
│   └── evaluate.py             # IR metrics calculation & benchmark
└── data/
    ├── the_batch_articles.json # Ingested articles dataset
    ├── images/                 # Downloaded figures and illustrations
    ├── chroma_db/              # Persistent ChromaDB vector collections
    ├── eval_results.json       # Benchmark evaluation output
    └── eval_metrics.png        # Generated evaluation chart
```
