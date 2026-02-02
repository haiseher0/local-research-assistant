import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class ResearchRAG:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        print("Initializing models...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatOllama(model="llama3", temperature=0)
        self.vector_store = None

    def ingest(self):
        if not os.path.exists(self.pdf_path):
            print(f"ERROR: File {self.pdf_path} not found!")
            return

        print(f"Loading {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()

        print("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        print("Building vector database...")
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        print("Done.")

    def query(self, question: str):
        if not self.vector_store:
            return "Error: Run ingest() first!"

        print(f"\nThinking on: {question}...")
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.invoke(question)

if __name__ == "__main__":
    rag = ResearchRAG(pdf_path=pdf_file)
    rag.ingest()

    # 2. Интерактивный цикл
    print("\n" + "="*50)
    print("🤖 Research Assistant Ready! (Type 'exit' to quit)")
    print("="*50)

    while True:
        user_input = input("\nUser: ")

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            print(rag.query(user_input))
        except Exception as e:
            print(f"Error: {e}")
