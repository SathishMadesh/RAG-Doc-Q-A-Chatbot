# 📄 RAG Document Q&A Chatbot

A **Retrieval-Augmented Generation (RAG) based chatbot** that allows users to upload PDF documents and ask questions based on their content. The system retrieves relevant context from the documents and generates accurate answers using an LLM.

---

## 🚀 Features

- 📂 Upload multiple PDF documents  
- 🔍 Semantic search using vector embeddings  
- 🧠 Context-aware question answering  
- ⚡ Fast retrieval using FAISS  
- 🖥️ Interactive UI with Streamlit  
- 🔒 Works locally with Ollama (no API limits)

---

## 🧠 Tech Stack

- **LLM:** Groq (LLaMA 3)  
- **Embeddings:** Ollama (`nomic-embed-text`)  
- **Framework:** LangChain  
- **Vector DB:** FAISS  
- **Frontend:** Streamlit  

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone <your-repo-link>
cd <project-folder> 

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Create a .env file and add:

GROQ_API_KEY=your_groq_api_key

### 5. Setup Ollama (for embeddings)
ollama serve
ollama pull nomic-embed-text

### 6. Run the Application
streamlit run app.py

### 🛠️ How It Works
User uploads PDF files
Documents are split into chunks
Embeddings are generated
Stored in FAISS vector database
User query → similarity search
Relevant context → passed to LLM
LLM generates final answer

### 📸 Usage
Upload PDF documents
Click "Document Embedding"
Ask questions in the input box
View answers and document context

### ⚠️ Notes
Ensure Ollama is running before starting the app
Large PDFs may take time to process
First run may be slower due to embedding generation
