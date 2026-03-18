# 🏥 Customer Support Agent – Medical RAG Chatbot

An AI-powered **Customer Support & Medical Assistant Chatbot** built using **Retrieval-Augmented Generation (RAG)**.  
This system provides **context-aware, safe, and explainable responses** to user queries related to symptoms, diseases, and general medical information.

---

## 🚀 Demo
  
📊 **Project PPT:** 👉 **https://docs.google.com/presentation/d/1NgRws7dhrqhGvBAUeFsTniaUGzF8SQJD/edit?usp=sharing&ouid=110464149473359827305&rtpof=true&sd=true**

---

## 📌 Overview

This project combines **LLMs + Vector Search + Medical Knowledge Base** to create an intelligent assistant that:

- Understands user symptoms/questions  
- Retrieves relevant medical knowledge  
- Generates safe, structured responses  
- Recommends consulting healthcare professionals  

---

## 🧠 Key Features

✨ **AI-Powered Medical Chatbot**  
- Natural language understanding for symptom-based queries  

🔍 **Retrieval-Augmented Generation (RAG)**  
- Uses vector search to fetch relevant medical context  

⚡ **FAISS Vector Database**  
- Fast similarity search for accurate responses  

🤖 **LLM Integration (Groq - LLaMA 3.3)**  
- Generates human-like, contextual answers  

📊 **Multi-Source Medical Data**
- MedQuAD dataset (Q&A)
- Disease & symptom datasets  

🌐 **Interactive Web UI**
- Clean chat interface using HTML/CSS  
- Real-time response handling  

🛡️ **Safe Medical Guidance**
- Avoids harmful suggestions  
- Encourages consulting professionals  

---

## 🏗️ Architecture
User Query
   ↓
Frontend (HTML Chat UI)
   ↓
Flask Backend API
   ↓
RAG Pipeline:
   - Embedding (Sentence Transformers)
   - FAISS Vector Search
   - Context Retrieval
   - Groq LLM Response
   ↓
Final Response → User
