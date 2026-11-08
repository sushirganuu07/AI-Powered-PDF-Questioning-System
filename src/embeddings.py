
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.document_loader import load_pdf
from src.text_splitter import split_documents

# Load environment variables
load_dotenv()


def generate_embeddings(chunks):

    # Create Gemini embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT"
    )

    # Convert text chunks into vectors
    texts = [chunk.page_content for chunk in chunks]

    vectors = embeddings.embed_documents(texts)

    return vectors


if __name__ == "__main__":

    # Load PDF
    documents = load_pdf("data/sample.pdf")

    # Split documents
    chunks = split_documents(documents)

    # Generate embeddings
    vectors = generate_embeddings(chunks)

    print("Total chunks:", len(chunks))
    print("Total embeddings:", len(vectors))
    print("Vector dimensions:", len(vectors[0]))

    print("\nFirst embedding:")
    print(vectors[0][:10])
