import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from groq import Groq
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_type' not in st.session_state:
    st.session_state.rag_type = None

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# API Key input
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Enter your Groq API key"
)

# Model selection
model_name = st.sidebar.selectbox(
    "Select Model",
    ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"],
    help="Choose the Groq model to use"
)

# RAG Type selection
rag_type = st.sidebar.radio(
    "Select RAG Type",
    ["PDF Upload", "Web Scraping"],
    help="Choose between PDF-based RAG or web scraping RAG"
)

st.sidebar.markdown("---")

# Main title
st.title("🤖 RAG Chatbot System")
st.markdown("Ask questions based on your documents or scraped web content!")

# Helper functions
@st.cache_resource
def load_embeddings():
    """Load the embedding model"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(pdf_file):
    """Process uploaded PDF file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        return docs
    finally:
        os.unlink(tmp_path)

def scrape_website(url):
    """Scrape content from a website"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        texts = []
        for tag in soup.find_all(["p", "li", "a", "td", "h1", "h2", "h3"]):
            text = tag.get_text(strip=True)
            if text and len(text) > 30:
                texts.append(text)
        
        full_text = "\n".join(texts)
        return [Document(page_content=full_text, metadata={"source": url})]
    except Exception as e:
        st.error(f"Error scraping website: {str(e)}")
        return None

def create_vector_store(documents):
    """Create FAISS vector store from documents"""
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_documents(documents)
    embeddings = load_embeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    return vector_db, len(chunks)

def chatbot(message, vector_db, api_key, model):
    """Generate response using RAG"""
    if not api_key:
        return "Please provide a Groq API key in the sidebar."
    
    # Retrieve relevant chunks
    results = vector_db.similarity_search(message, k=3)
    context = []
    
    for i in results:
        context.append(f"Chunk: {i.page_content}")
    
    # Prepare prompt
    system_prompt = """
You are a smart chatbot. Respond to user questions only by referring to the data in the knowledge base.
Provide well-structured responses without mentioning the chunks you're referencing.
If the information is not in the knowledge base, say so clearly.
"""
    
    final_prompt = f"{system_prompt}\n\nKnowledge Base: {context}"
    
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0,
            max_completion_tokens=1024
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # PDF Upload Section
    if rag_type == "PDF Upload":
        st.subheader("📄 PDF Upload")
        uploaded_file = st.file_uploader(
            "Upload a PDF file",
            type=['pdf'],
            help="Upload a PDF document to create a knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    docs = process_pdf(uploaded_file)
                    vector_db, num_chunks = create_vector_store(docs)
                    st.session_state.vector_db = vector_db
                    st.session_state.rag_type = "PDF"
                    st.session_state.chat_history = []
                    st.success(f"✅ PDF processed! Created {num_chunks} chunks.")
    
    # Web Scraping Section
    else:
        st.subheader("🌐 Web Scraping")
        url = st.text_input(
            "Enter Website URL",
            placeholder="https://example.com",
            help="Enter the URL of the website to scrape"
        )
        
        if url and st.button("Scrape Website", type="primary"):
            with st.spinner("Scraping website..."):
                docs = scrape_website(url)
                if docs:
                    vector_db, num_chunks = create_vector_store(docs)
                    st.session_state.vector_db = vector_db
                    st.session_state.rag_type = "Web"
                    st.session_state.chat_history = []
                    st.success(f"✅ Website scraped! Created {num_chunks} chunks.")

with col2:
    st.subheader("ℹ️ Status")
    if st.session_state.vector_db is not None:
        st.success("✅ Knowledge base ready!")
        st.info(f"Type: {st.session_state.rag_type}")
    else:
        st.warning("⚠️ No knowledge base loaded")
    
    if st.button("Clear Knowledge Base"):
        st.session_state.vector_db = None
        st.session_state.chat_history = []
        st.session_state.rag_type = None
        st.rerun()

st.markdown("---")

# Chat interface
st.subheader("💬 Chat")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if st.session_state.vector_db is None:
        st.error("Please upload a PDF or scrape a website first!")
    elif not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar!")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot(
                    prompt,
                    st.session_state.vector_db,
                    groq_api_key,
                    model_name
                )
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>Built with Streamlit, LangChain, FAISS, and Groq</small>
    </div>
    """,
    unsafe_allow_html=True
)