from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from src.document_loader import load_multiple_pdfs
from src.text_splitter import split_documents

load_dotenv()


def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_name="multi_document_collection"
    )

    return vector_store


if __name__ == "__main__":
    documents = load_multiple_pdfs("data")
    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    print("\nTotal documents:", len(documents))
    print("Total chunks stored:", len(chunks))
    print("All PDFs stored successfully in ChromaDB!")