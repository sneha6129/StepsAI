Advanced Textbook Question Answering System:
This repository contains a Streamlit-based web application designed to extract content from textbooks, create hierarchical structures, and provide advanced question-answering capabilities using modern NLP techniques.

Features
Asynchronous Content Extraction: Efficiently extracts text from PDF files.
Hierarchical Tree-based Indexing: Organizes textbook content into a hierarchical structure for better retrieval.
SBERT for Semantic Search: Utilizes Sentence-BERT for effective semantic search.
Question Answering with Hugging Face Models: Provides both extractive and generative question-answering capabilities.
Streamlit User Interface: Offers an interactive UI for uploading textbooks and querying content.

Installation
Ensure you have Python 3.7 or above installed. Install the required dependencies using the following command:

pip install streamlit anytree sentence-transformers nltk PyPDF2 transformers torch
To run the app:
streamlit run app.py


Required Libraries
streamlit: For creating the web application interface.
anytree: For handling hierarchical data structures.
sentence-transformers: For embedding text using Sentence-BERT.
nltk: For natural language processing tasks such as tokenization.
PyPDF2: For extracting text from PDF files.
transformers: For using pre-trained NLP models from Hugging Face.
torch: Required by the transformers library for model inference.

The steps involved in this are:
1.Improved Content Extraction with Asynchronous Processing
2.Enhanced Hierarchical Tree-based Indexing
3.Improved Retrieval Techniques using SBERT
4.Enhanced Multi-document/Topic/Section-based RAG
5.Hugging Face-powered Question Answering with Generation
6.Helper Function for Asynchronous PDF Processing
7.Enhanced Main Function and User Interface

Summary:
This repository contains a Streamlit-based web application designed to extract content from textbooks, create hierarchical structures, and provide advanced question-answering capabilities using modern NLP techniques.
