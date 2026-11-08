from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def create_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="RETRIEVAL_QUERY"
    )

    vector_store = Chroma(
        collection_name="multi_document_collection",
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    return vector_store


def search_documents(question):
    vector_store = create_retriever()

    results = vector_store.similarity_search(
        question,
        k=3
    )

    return results


if __name__ == "__main__":
    questions = [
        "What is supervised learning?",
        "What is Python?",
        "What is Generative AI?"
    ]

    for question in questions:
        print("\nQuestion:", question)

        results = search_documents(question)

        for i, doc in enumerate(results):
            print(f"\nResult {i + 1}")
            print("Source:", doc.metadata.get("source"))
            print("Page:", doc.metadata.get("page", 0) + 1)
            print("Content:", doc.page_content[:200])