from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.document_loader import load_multiple_pdfs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    documents = load_multiple_pdfs("data")

    chunks = split_documents(documents)

    print("\nTotal PDF pages:", len(documents))
    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}")
        print("Source:", chunk.metadata.get("source"))
        print("Page:", chunk.metadata.get("page", 0) + 1)
        print("Content:", chunk.page_content[:150])