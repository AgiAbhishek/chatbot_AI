import streamlit as st
import tempfile
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import json

# Import components
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.llm_handler import LLMHandler
from src.rag_pipeline import RAGPipeline
from src.utils import setup_directories, format_file_size

def initialize_components():
    """Initialize the RAG pipeline components"""
    try:
        # Setup directories
        setup_directories()
        
        # Initialize components
        document_processor = DocumentProcessor()
        vector_store = VectorStore()
        llm_handler = LLMHandler()
        
        # Create RAG pipeline
        rag_pipeline = RAGPipeline(
            document_processor=document_processor,
            vector_store=vector_store,
            llm_handler=llm_handler
        )
        
        return rag_pipeline
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None

def get_existing_documents():
    """Get list of documents already in data folder"""
    data_folder = Path("data")
    documents = []
    
    if data_folder.exists():
        for file_path in data_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt']:
                # Check if already processed
                chunks_file = Path("chunks") / f"{file_path.stem}_chunks.json"
                is_processed = chunks_file.exists()
                
                documents.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'processed': is_processed,
                    'type': file_path.suffix.lower()
                })
    
    return documents

def process_document_from_path(file_path, rag_pipeline):
    """Process document from file path"""
    try:
        chunks = rag_pipeline.process_document(file_path)
        
        # Update session state
        st.session_state.document_processed = True
        st.session_state.chunk_count = len(chunks)
        st.session_state.current_document = Path(file_path).name
        
        return True, chunks
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return False, []

def process_uploaded_document(uploaded_file, rag_pipeline):
    """Process uploaded document and create vector store"""
    try:
        if uploaded_file is not None:
            # Save to data folder first
            data_folder = Path("data")
            data_folder.mkdir(exist_ok=True)
            
            file_path = data_folder / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process document
            success, chunks = process_document_from_path(str(file_path), rag_pipeline)
            
            return success
            
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return False

def display_chat_message(message: Dict[str, Any]):
    """Display a chat message with proper formatting"""
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📖 Sources", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.write(f"**Source {i}:**")
                    st.write(f"*Score: {source['score']:.3f}*")
                    st.write(source["content"])
                    st.write("---")

def main():
    st.title("🤖 AI Document Chatbot")
    st.markdown("Ask questions about your documents and get answers with source citations!")
    
    # Initialize session state
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Initialize RAG pipeline if not already done
        if st.session_state.rag_pipeline is None:
            with st.spinner("Initializing system..."):
                st.session_state.rag_pipeline = initialize_components()
        
        if st.session_state.rag_pipeline is None:
            st.error("Failed to initialize system. Please refresh the page.")
            return
        
        st.divider()
        
        # Document Library
        st.header("📚 Document Library")
        
        # Get existing documents
        existing_docs = get_existing_documents()
        
        if existing_docs:
            st.subheader("Available Documents")
            
            # Create tabs for processed and unprocessed
            processed_docs = [doc for doc in existing_docs if doc['processed']]
            unprocessed_docs = [doc for doc in existing_docs if not doc['processed']]
            
            if processed_docs:
                st.write("**Ready to Chat:**")
                for doc in processed_docs:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        if st.button(f"📄 {doc['name']}", key=f"select_{doc['name']}"):
                            # Load this document
                            success, chunks = process_document_from_path(doc['path'], st.session_state.rag_pipeline)
                            if success:
                                st.success(f"✅ Loaded: {doc['name']}")
                                st.rerun()
                    with col2:
                        st.write(f"{format_file_size(doc['size'])}")
                    with col3:
                        st.write("✅ Ready")
            
            if unprocessed_docs:
                st.write("**Need Processing:**")
                for doc in unprocessed_docs:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        if st.button(f"🔄 Process {doc['name']}", key=f"process_{doc['name']}"):
                            with st.spinner(f"Processing {doc['name']}..."):
                                success, chunks = process_document_from_path(doc['path'], st.session_state.rag_pipeline)
                                if success:
                                    st.success(f"✅ Processed: {doc['name']}")
                                    st.rerun()
                    with col2:
                        st.write(f"{format_file_size(doc['size'])}")
                    with col3:
                        st.write("⏳ Pending")
        else:
            st.info("No documents found in data/ folder")
        
        st.divider()
        
        # Upload new document
        st.header("📤 Upload New Document")
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['txt', 'pdf', 'docx'],
            help="Upload a document to chat with (10,000+ words recommended)"
        )
        
        if uploaded_file is not None:
            # Check if this is a new document
            if not hasattr(st.session_state, 'current_document') or st.session_state.current_document != uploaded_file.name:
                with st.spinner("Processing document..."):
                    success = process_uploaded_document(uploaded_file, st.session_state.rag_pipeline)
                    
                if success:
                    st.success(f"✅ Document processed: {uploaded_file.name}")
                    st.info(f"Created {st.session_state.chunk_count} chunks")
                    st.rerun()
                else:
                    st.error("❌ Failed to process document")
            else:
                st.success(f"✅ Document loaded: {uploaded_file.name}")
                st.info(f"Chunks: {st.session_state.chunk_count}")
        
        st.divider()
        
        # Current Document Status
        if st.session_state.document_processed and hasattr(st.session_state, 'current_document'):
            st.header("📋 Current Session")
            st.info(f"**Active Document:** {st.session_state.current_document}")
            st.info(f"**Chunks:** {st.session_state.chunk_count}")
            
            if st.button("🔄 Reset Session"):
                st.session_state.rag_pipeline.reset()
                st.session_state.document_processed = False
                st.session_state.messages = []
                st.rerun()
        
        # System stats
        if st.session_state.rag_pipeline:
            st.header("📈 System Stats")
            stats = st.session_state.rag_pipeline.get_pipeline_stats()
            st.metric("Available Documents", len(existing_docs))
            if st.session_state.document_processed:
                st.metric("Active Chunks", stats.get('total_chunks', 0))
                st.metric("Vector Dimensions", stats.get('vector_dimensions', 0))
    
    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(message)
    
    # Check if document is processed
    if not st.session_state.document_processed:
        st.info("Please upload a document or select one from the library to start chatting.")
        st.stop()
    
    # User input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            try:
                # Get response from RAG pipeline
                response_data = st.session_state.rag_pipeline.get_response(prompt)
                
                if response_data and "response" in response_data:
                    # Get the response text (fixed generator issue)
                    response_text = response_data["response"]
                    
                    # Handle both string and generator responses
                    if isinstance(response_text, str):
                        message_placeholder.markdown(response_text)
                    else:
                        # Handle generator case
                        full_response = ""
                        for chunk in response_text:
                            if chunk:
                                full_response += str(chunk)
                        message_placeholder.markdown(full_response)
                        response_text = full_response
                    
                    # Display sources section
                    if response_data["sources"]:
                        with sources_placeholder.expander("📄 Document Sources", expanded=False):
                            for i, source in enumerate(response_data["sources"], 1):
                                st.write(f"**Source {i}** (Relevance: {source['score']:.2f})")
                                st.write(source["content"])
                                if i < len(response_data["sources"]):
                                    st.write("---")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": response_data["sources"]
                    })
                else:
                    error_msg = "Sorry, I couldn't generate a response. Please try again."
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    import streamlit as st
    main()
