# 📄 pdf-query-backend

A simple backend service that enables **question answering over PDF documents**.  
Built with **Express.js**, **LangChain**, and **OpenAI API**.

## 🚀 Features

- Upload a PDF file and generate embeddings
- Split documents into chunks for efficient retrieval
- Store vectors in an in-memory vector store (demo purpose)
- Query with natural language and get concise answers powered by GPT
- Retrieval-Augmented Generation (RAG) pipeline using LangChain

## 🛠️ Tech Stack

- **Backend**: Node.js + Express.js
- **AI / NLP**: LangChain, OpenAI GPT-3.5 Turbo, OpenAI Embeddings
- **Vector Store**: MemoryVectorStore (can be replaced with FAISS, Pinecone, etc.)
- **File Upload**: Multer