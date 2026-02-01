import os
import sys
from typing import List

# Импорты для бесплатного стека
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Бесплатные эмбеддинги
from langchain_community.chat_models import ChatOllama   # Бесплатная LLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

class ResearchRAG:
    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 1. Используем бесплатные эмбеддинги от HuggingFace (работают на CPU)
        print("Initializing Embeddings Model (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.vector_store = None
        self.chain = None

    def ingest(self) -> None:
        """Loads PDF, splits it, and builds the local Vector Index."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"File {self.pdf_path} not found.")

        print(f"Loading document: {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        print(f"Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        print(f"Created {len(texts)} chunks.")

        print("Building FAISS index (this runs locally)...")
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        print("Index built successfully.")

    def setup_chain(self) -> None:
        """Configures the QA chain using local Ollama model."""
        if not self.vector_store:
            raise ValueError("Run ingest() first.")

        # 2. Используем локальную Llama 3 через Ollama
        # Убедись, что ты запустил 'ollama run llama3' в терминале до этого
        llm = ChatOllama(model="llama3", temperature=0)
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    def query(self, question: str) -> str:
        if not self.chain:
            self.setup_chain()

        print(f"\nThinking on: {question}...")
        try:
            response = self.chain.invoke({"query": question})
            answer = response["result"]
            sources = response["source_documents"]
            
            # Собираем страницы для цитирования
            pages = sorted(list(set([doc.metadata.get("page", 0) + 1 for doc in sources])))
            return f"Answer: {answer}\n\nSources found on pages: {pages}"
        except Exception as e:
            return f"Error during inference. Make sure Ollama is running! Details: {e}"

# --- Usage ---
if __name__ == "__main__":
    # Укажи имя своего PDF файла здесь
    pdf_file = "paper.pdf"
    
    rag = ResearchRAG(pdf_path=pdf_file)
    
    try:
        rag.ingest()
        
        # Пример вопросов
        print(rag.query("What is the main mathematical idea of this paper?"))
        print(rag.query("Summarize the conclusion."))
        
    except Exception as e:
        print(f"Critical Error: {e}")