# AI Powered PDF Questioning System

An AI-powered application that allows users to upload multiple PDF documents and ask questions about their content. The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate context-aware answers using Google Gemini.

## Features

- Upload multiple PDF files.
- Extract and process PDF content.
- Split documents into smaller text chunks.
- Generate embeddings using Google Gemini.
- Store and retrieve document embeddings using ChromaDB.
- Ask questions about uploaded PDFs.
- Get AI-generated answers based on document content.
- View source PDF filenames and page numbers.
- Interactive Streamlit chat interface.
- Chat history and clear chat functionality.

## Tech Stack

- Python
- LangChain
- Google Gemini API
- ChromaDB
- Streamlit
- PyPDF

## Project Structure

```text
task2/
├── .streamlit/
├── data/
├── src/
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── rag_pipeline.py
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd task2
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own Gemini API key. Never upload your `.env` file to GitHub.

### 6. Run the application

```bash
streamlit run app.py
```

## How It Works

1. User uploads PDF documents.
2. The application extracts text from PDFs.
3. Text is divided into smaller chunks.
4. Gemini embeddings convert chunks into vector representations.
5. ChromaDB stores the vectors.
6. The retriever finds relevant chunks for the user's question.
7. Google Gemini generates an answer using the retrieved context.
8. The application displays the answer with source information.

## Author

Ganesh Sushir

## Project Type

Python Internship Project – Task 2