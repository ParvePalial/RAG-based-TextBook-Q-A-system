Smart Document Q&A System
Overview
The Smart Document Q&A System is a Retrieval-Augmented Generation (RAG) application designed to ingest large-scale PDF documents and provide highly accurate, context-aware answers to user queries. By leveraging semantic vector search and advanced Large Language Models, this system eliminates AI hallucinations by grounding responses strictly in the provided document text.

Architecture & Development Lifecycle
This project is engineered with a modular Software Development Life Cycle (SDLC) in mind, dividing the system into three distinct, decoupled components. This architecture supports parallel development, testing, and deployment:

Frontend Interface (app.py & templates/): A Flask-based web server rendering a responsive, modern UI for document upload and query execution.

Data Ingestion & Pipeline (utils.py & embeddings.py): Handles the extraction of text from dense PDFs, applies an overlapping sliding-window chunking algorithm, and generates high-dimensional vector embeddings.

AI Retrieval Engine (retriever.py): Orchestrates semantic search against a local vector database and constructs strict prompt contexts for the LLM.

Features
Large Document Processing: Capable of ingesting 1000+ page PDFs efficiently.

Semantic Search: Utilizes FAISS (Facebook AI Similarity Search) for rapid, mathematical text retrieval.

Overlapping Context Windows: Implements an 800-word chunking strategy with a 100-word overlap to preserve contextual continuity across paragraph boundaries.

Local Vector Storage: Runs the embedding model (all-MiniLM-L6-v2) and FAISS locally to minimize API costs and latency.

Gemini Flash Integration: Uses Google's gemini-2.5-flash model for high-speed, accurate text generation constrained entirely to the source material.

Prerequisites
Python 3.8 or higher

Git

Installation
Clone the repository

Bash
git clone https://github.com/yourusername/smart-document-qa.git
cd smart-document-qa
Create a virtual environment (Recommended)

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies

Bash
pip install -r requirements.txt
Configure Environment Variables
Create a .env file in the root directory and add your Google Gemini API key. Ensure this file is added to your .gitignore.

Plaintext
GEMINI_API_KEY=your_api_key_here
Usage
Start the Flask application server:

Bash
python app.py
Navigate to http://127.0.0.1:5000 in your web browser.

Upload a target PDF document to initialize the vector database.

Submit natural language queries to retrieve contextually accurate answers based solely on the uploaded text.

Project Structure
Plaintext
smart-document-qa/
├── app.py                 # Flask server and application routing
├── retriever.py           # FAISS search and Gemini generation logic
├── embeddings.py          # Vector embedding generation and FAISS storage
├── utils.py               # PDF extraction and text chunking logic
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (ignored by Git)
├── .gitignore             # Git ignore configurations
└── templates/
    └── index.html         # Frontend user interface
Contributors
[Your Name/Handle] - Frontend & Routing

[Collaborator 1 Name] - Data Pipeline & Vectorization

[Collaborator 2 Name] - AI Integration & Prompt Engineering

License
Distributed under the MIT License. See LICENSE for more information.
