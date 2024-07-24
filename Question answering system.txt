import io
import re
import uuid
import asyncio
import concurrent.futures
from typing import List, Dict, Any
from anytree import Node, RenderTree
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize, sent_tokenize
import PyPDF2
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import sentencepiece

# Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    import nltk
    nltk.download('punkt', quiet=True)

download_nltk_data()

# Step 1: Improved Content Extraction with Asynchronous Processing
async def extract_content_from_pdf_async(pdf_file: io.BytesIO) -> str:
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, extract_content_from_pdf_sync, pdf_file)

def extract_content_from_pdf_sync(pdf_file: io.BytesIO) -> str:
    content = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            content += page.extract_text() + "\n"
        return content
    except Exception as e:
        st.error(f"Error extracting PDF content: {str(e)}")
        return ""

# Step 2: Enhanced Hierarchical Tree-based Indexing
class TextbookNode(Node):
    def __init__(self, name, parent=None, content="", node_type="", node_id=None):
        super().__init__(name, parent=parent)
        self.content = content
        self.node_type = node_type
        self.node_id = node_id or str(uuid.uuid4())

def node_to_dict(node: TextbookNode) -> dict:
    return {
        "name": node.name,
        "content": node.content,
        "node_type": node.node_type,
        "node_id": node.node_id,
        "children": [node_to_dict(child) for child in node.children]
    }

def dict_to_node(node_dict: dict, parent=None) -> TextbookNode:
    node = TextbookNode(
        name=node_dict["name"],
        parent=parent,
        content=node_dict["content"],
        node_type=node_dict["node_type"],
        node_id=node_dict["node_id"]
    )
    for child_dict in node_dict["children"]:
        dict_to_node(child_dict, node)
    return node

@st.cache_data
def create_hierarchical_tree(content: str, book_title: str) -> dict:
    root = TextbookNode(book_title, node_type="book")
    lines = content.split('\n')
    current_chapter = None
    current_section = None
    
    chapter_pattern = re.compile(r'^Chapter\s+\d+', re.IGNORECASE)
    section_pattern = re.compile(r'^\d+\.\d+\s+', re.IGNORECASE)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if chapter_pattern.match(line):
            current_chapter = TextbookNode(line, parent=root, node_type="chapter")
            current_section = None
        elif section_pattern.match(line):
            if current_chapter:
                current_section = TextbookNode(line, parent=current_chapter, node_type="section")
        elif current_section:
            current_section.content += line + "\n"
        elif current_chapter:
            current_chapter.content += line + "\n"
    
    return node_to_dict(root)

# Step 3: Improved Retrieval Techniques using SBERT
@st.cache_resource
def create_sbert_index(documents: List[str]):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(documents, convert_to_tensor=True)
    return model, embeddings

def sbert_retrieval(query: str, model, embeddings, documents: List[str], top_k=10) -> List[int]:
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]
    return [hit['corpus_id'] for hit in hits]

# Step 4: Enhanced Multi-document/Topic/Section-based RAG
@st.cache_data
def prepare_retrieval_data(_textbooks):
    all_sections = []
    section_metadata = []
    
    for textbook in _textbooks:
        for node in textbook.descendants:
            if node.content:
                all_sections.append(node.content)
                section_metadata.append({
                    "book": textbook.name,
                    "chapter": node.parent.name if node.node_type == "section" else node.name if node.node_type == "chapter" else "",
                    "section": node.name if node.node_type == "section" else "",
                    "node_id": node.node_id
                })
    
    return all_sections, section_metadata

# Step 5: Hugging Face-powered Question Answering with Generation
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@st.cache_resource
def load_generation_model():
    model_name = "t5-small"  # You can choose any other generative model as well
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def answer_question_with_huggingface(query: str, context: List[Dict[str, Any]]) -> str:
    qa_pipeline = load_qa_pipeline()
    combined_context = " ".join([item['content'] for item in context])
    result = qa_pipeline(question=query, context=combined_context)
    return result['answer']

def generate_answer_with_t5(query: str, contexts: List[Dict[str, Any]]) -> str:
    model, tokenizer = load_generation_model()
    combined_context = " ".join([item['content'] for item in contexts])
    input_text = f"question: {query} context: {combined_context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids, max_length=150, num_return_sequences=1, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Helper function for asynchronous PDF processing
async def process_books(book_files):
    loop = asyncio.get_event_loop()
    tasks = [extract_content_from_pdf_async(book) for book in book_files]
    return await asyncio.gather(*tasks)

# Step 6: Enhanced Main function and User Interface
async def main_async():
    st.title("Advanced Textbook Question Answering System")
    
    st.sidebar.title("Textbook Selection")
    book1 = st.sidebar.file_uploader("Upload Textbook 1 (PDF)", type="pdf")
    book2 = st.sidebar.file_uploader("Upload Textbook 2 (PDF)", type="pdf")
    book3 = st.sidebar.file_uploader("Upload Textbook 3 (PDF)", type="pdf")
    
    if book1 and book2 and book3:
        with st.spinner("Processing textbooks... This may take a moment."):
            textbook_contents = await process_books([book1, book2, book3])
            
            textbook_trees = [
                dict_to_node(create_hierarchical_tree(content, f"Textbook {i+1}")) 
                for i, content in enumerate(textbook_contents)
            ]
            
            all_sections, section_metadata = prepare_retrieval_data(textbook_trees)
            
            if all_sections:
                sbert_model, sbert_embeddings = create_sbert_index(all_sections)
                st.success("Textbooks loaded and indexed successfully!")
            else:
                st.error("No sections found in the uploaded textbooks.")
    
    query = st.text_input("Enter your question:")
    
    if query and book1 and book2 and book3:
        with st.spinner("Searching for relevant information..."):
            relevant_content = []
            sbert_results = sbert_retrieval(query, sbert_model, sbert_embeddings, all_sections)
            
            for idx in sbert_results:
                relevant_content.append({
                    "content": all_sections[idx],
                    "metadata": section_metadata[idx]
                })
            
            if relevant_content:
                # Use the generative model to answer the question based on retrieved contexts
                answer = generate_answer_with_t5(query, relevant_content)
                st.write("### Answer:")
                st.write(answer)
            else:
                st.write("No relevant information found in the uploaded textbooks.")

if __name__ == "__main__":
    asyncio.run(main_async())
