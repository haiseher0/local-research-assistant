# Local Research Assistant (RAG Pipeline) 🧠

An offline, privacy-focused AI assistant for analyzing scientific papers.
Built using **LangChain**, **FAISS** (Vector DB), and **Llama 3** (via Ollama).

## Features
- **100% Local Inference:** No data is sent to OpenAI or cloud API. Perfect for sensitive documents.
- **RAG Architecture:** Uses Retrieval-Augmented Generation to ground LLM responses in facts.
- **Citation:** The model cites the exact page numbers used to generate the answer.

## Tech Stack
- Language: Python 3.10+
- LLM: Meta Llama 3 (Quantized) via Ollama
- Embeddings: HuggingFace (`all-MiniLM-L6-v2`)
- Vector Store: FAISS (CPU optimized)

## How to Run

1. Clone the repo:
   ```bash
   git clone [https://github.com/haiseher0/local-research-assistant.git](https://github.com/YOUR_USERNAME/local-research-assistant.git)
   cd local-research-assistant

2. Install dependencies:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Setup Ollama:

Install Ollama

Pull the model: ollama run llama3

4. Run:

Place your target PDF in the folder and rename it to paper.pdf.

Run the script:

python main.py