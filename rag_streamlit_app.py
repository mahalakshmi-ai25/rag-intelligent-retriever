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
    value=default.apikey
    help="Enter your Groq API key"
)

model_name = st.sidebar.selectbox(
    "Model",
    ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
    index=0
)

# RAG Type Selection
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Select RAG Type")
rag_option = st.sidebar.radio(
    "Choose your data source:",
    ["PDF Upload", "Web Scraping"],
    index=0
)

# Main title
st.title("🤖 RAG-Powered Chatbot System")
st.markdown("Ask questions based on your uploaded documents or scraped web content!")

# Function definitions
@st.cache_resource
def load_embeddings():
    """Load HuggingFace embeddings model"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    """Process uploaded PDF file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return docs
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def create_vector_db(documents, embeddings):
    """Create FAISS vector database from documents"""
    try:
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        vector_db = FAISS.from_documents(chunks, embeddings)
        return vector_db, len(chunks)
    except Exception as e:
        st.error(f"Error creating vector database: {str(e)}")
        return None, 0

def scrape_website(url):
    """Scrape website content"""
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
        
        document = Document(
            page_content=full_text,
            metadata={"source": url}
        )
        
        return [document]
    except Exception as e:
        st.error(f"Error scraping website: {str(e)}")
        return None

def chatbot(message, vector_db, api_key, model):
    """RAG chatbot function"""
    try:
        # Retrieve relevant chunks
        result = vector_db.similarity_search(message, k=3)
        context = []
        for i in result:
            context.append(f"Chunk: {i.page_content}")
        
        # Create Groq client
        client = Groq(api_key=api_key)
        
        prompt = """
You are a smart chatbot. You need to respond to user questions only by referring to the data present within the below knowledge base.
Don't give any reference to the chunk which you are referring to. Just return a well-structured response as the answer to the user question.
"""
        
        finalprompt = f"{prompt}\n\nKnowledge Base: {context}"
        
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": finalprompt
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            temperature=0,
            max_completion_tokens=1024
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📂 Data Source Setup")
    
    if rag_option == "PDF Upload":
        st.markdown("### Upload PDF Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to create your knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    # Load embeddings
                    embeddings = load_embeddings()
                    
                    # Process PDF
                    docs = process_pdf(uploaded_file)
                    
                    if docs:
                        # Create vector database
                        vector_db, num_chunks = create_vector_db(docs, embeddings)
                        
                        if vector_db:
                            st.session_state.vector_db = vector_db
                            st.session_state.rag_type = "PDF"
                            st.session_state.chat_history = []
                            st.success(f"✅ PDF processed successfully! Created {num_chunks} chunks.")
                            st.info(f"📄 Document: {uploaded_file.name}")
    
    else:  # Web Scraping
        st.markdown("### Enter Website URL")
        url = st.text_input(
            "Website URL",
            value="https://www.icmr.gov.in/tenders",
            help="Enter the URL of the website you want to scrape"
        )
        
        if st.button("🌐 Scrape Website", type="primary"):
            with st.spinner(f"Scraping {url}..."):
                # Load embeddings
                embeddings = load_embeddings()
                
                # Scrape website
                documents = scrape_website(url)
                
                if documents:
                    # Create vector database
                    vector_db, num_chunks = create_vector_db(documents, embeddings)
                    
                    if vector_db:
                        st.session_state.vector_db = vector_db
                        st.session_state.rag_type = "Web"
                        st.session_state.chat_history = []
                        st.success(f"✅ Website scraped successfully! Created {num_chunks} chunks.")
                        st.info(f"🌐 Source: {url}")

with col2:
    st.subheader("💬 Chat Interface")
    
    if st.session_state.vector_db is None:
        st.info("👈 Please upload a PDF or scrape a website to start chatting!")
    else:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant"):
                    st.write(chat["answer"])
        
        # Chat input
        user_question = st.chat_input("Ask a question about your document...")
        
        if user_question:
            # Add user message to chat
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chatbot(
                        user_question,
                        st.session_state.vector_db,
                        groq_api_key,
                        model_name
                    )
                    st.write(response)
            
            # Save to chat history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": response
            })
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
if st.session_state.vector_db:
    st.sidebar.success(f"✅ {st.session_state.rag_type} RAG Active")
else:
    st.sidebar.warning("⚠️ No Knowledge Base Loaded")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

if st.sidebar.button("🔄 Reset All"):
    st.session_state.vector_db = None
    st.session_state.chat_history = []
    st.session_state.rag_type = None
    st.rerun()
