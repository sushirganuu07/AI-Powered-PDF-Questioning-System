from langchain_google_genai import ChatGoogleGenerativeAI
from src.retriever import search_documents
import os


def generate_answer(question):

    documents = search_documents(question)

    context = ""

    for doc in documents:
        page = doc.metadata.get("page", 0) + 1
        source = os.path.basename(doc.metadata.get("source", "Unknown PDF"))

        context += f"""
        Source: {source}
        Page: {page}
        Content: {doc.page_content}
        """

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    prompt = f"""
    You are a helpful PDF Question Answering Assistant.

    Answer using ONLY the context provided below.

    If the answer is not available in the context,
    say:
    "I could not find this information in the uploaded documents."

    Mention the PDF file name and page number in your answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    return response.text


if __name__ == "__main__":

    print("===== MULTIPLE PDF QUESTION ANSWERING SYSTEM =====")

    while True:

        question = input("\nEnter your question (or type 'exit' to stop): ")

        if question.lower() == "exit":
            print("Thank you for using PDF Q&A Assistant!")
            break

        if not question.strip():
            print("Please enter a valid question.")
            continue

        answer = generate_answer(question)

        print("\nGenerated Answer:")
        print(answer)