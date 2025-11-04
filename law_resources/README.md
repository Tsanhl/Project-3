# Law Resources Knowledge Base

This folder contains your personal law resources (journals, notes, cases) that will be used as a knowledge base for the RAG (Retrieval-Augmented Generation) system.

## Folder Structure

- `journals/` - Place law journal articles, academic papers, and scholarly articles here
- `notes/` - Place your personal law notes, study materials, and summaries here
- `cases/` - Place case law documents, case summaries, and legal precedents here

## Supported File Formats

- PDF (`.pdf`)
- Text files (`.txt`, `.md`)
- Word documents (`.docx`, `.doc`)

## How It Works

The RAG system will:
1. Load all documents from this folder and subfolders
2. Split them into chunks for efficient searching
3. Create embeddings using sentence transformers
4. Build a FAISS vector store for fast similarity search
5. Retrieve relevant documents when answering law questions

## Usage

Simply place your law resources in the appropriate subfolders. The system will automatically:
- Index them on first use
- Retrieve relevant content when answering questions
- Combine with web search and free law databases for comprehensive answers

## Best Practices

1. Use descriptive filenames
2. Organize documents by topic in subfolders
3. Ensure text is extractable (scanned PDFs may need OCR)
4. Keep documents up-to-date for accuracy

The knowledge base is automatically initialized when you ask a law question!
