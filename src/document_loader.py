
import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):
    loader = PyPDFLoader(file_path, mode="page")
    documents = loader.load()
    return documents


def load_multiple_pdfs(folder_path):

    all_documents = []

    for file_name in os.listdir(folder_path):

        if file_name.lower().endswith(".pdf"):

            file_path = os.path.join(folder_path, file_name)

            print(f"Loading: {file_name}")

            documents = load_pdf(file_path)

            all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":

    documents = load_multiple_pdfs("data")

    print("\nTotal PDF pages loaded:", len(documents))

    print("\nPDF files loaded successfully!")
