# AI Document Chatbot with RAG

A sophisticated AI-powered document chatbot built with Streamlit that uses Retrieval-Augmented Generation (RAG) to answer questions from long documents with accurate source citations and streaming responses.

## Project Architecture

### System Overview
The application follows a modular RAG architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │    │   Processing     │    │   Vector Store  │
│  (PDF/DOCX/TXT) │───▶│   Pipeline       │───▶│   (FAISS DB)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   RAG Pipeline   │    │   LLM Handler   │
│   Interface     │◀──▶│   Orchestrator   │◀──▶│  (Groq/OpenAI)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow Architecture
1. **Document Ingestion**: User uploads or manually adds documents to `data/` folder
2. **Text Extraction**: PyPDF2/python-docx extracts raw text from documents
3. **Preprocessing**: NLTK tokenization, text cleaning, and normalization
4. **Semantic Chunking**: Intelligent text splitting into 100-300 word chunks with context preservation
5. **Embedding Generation**: SentenceTransformers (all-MiniLM-L6-v2) creates 384-dimensional vectors
6. **Vector Storage**: FAISS index stores embeddings with metadata for fast similarity search
7. **Query Processing**: User questions are embedded and matched against document chunks
8. **Context Retrieval**: Top-K similar chunks retrieved with relevance scoring
9. **Response Generation**: Groq LLaMA3-8B generates comprehensive 250-300 word responses
10. **Streaming Display**: Real-time response streaming with source citations

## Features

- **Multiple Document Formats**: Support for PDF, DOCX, and TXT files
- **Document Library**: View and manage manually added files in the `data/` folder
- **Intelligent Chunking**: Semantic text splitting with 100-300 word chunks
- **Vector Search**: FAISS-powered similarity search with sentence embeddings
- **AI Response Generation**: Groq LLaMA3-8B integration for comprehensive 250-300 word responses
- **Streaming Responses**: Real-time response generation with live text display
- **Source Citations**: Detailed source references with relevance scores
- **Dual Upload Methods**: Upload via UI or manually add to data folder

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-document-chatbot

# Install dependencies
pip install -r requirements.txt

# Or using uv (recommended)
uv sync
```

### 2. Set up API Keys

Create a `.env` file or set environment variables:

```bash
# For Groq API (recommended)
GROQ_API_KEY=your_groq_api_key_here

# Or for OpenAI API (alternative)
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run Preprocessing and Build RAG Pipeline

The system automatically handles preprocessing when you add documents:

#### Preprocessing Steps:
```bash
# Option 1: Automatic preprocessing via web interface
streamlit run app.py --server.port 5000

# Option 2: Manual preprocessing for data/ folder files
python -c "
from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.llm_handler import LLMHandler

# Initialize components
doc_processor = DocumentProcessor()
vector_store = VectorStore()
llm_handler = LLMHandler()
pipeline = RAGPipeline(doc_processor, vector_store, llm_handler)

# Process document (replace with your file)
chunks = pipeline.process_document('data/your_document.pdf')
print(f'Created {len(chunks)} chunks and embeddings')
"
```

#### Embedding Creation Process:
1. **Text Extraction**: Documents parsed and text extracted
2. **Chunk Generation**: Text split into semantic chunks (100-300 words)
3. **Embedding Generation**: SentenceTransformers creates 384-dimensional vectors
4. **Vector Storage**: FAISS index built and saved to `vectordb/`
5. **Metadata Storage**: Chunk content and metadata saved for retrieval

### 4. Run the Chatbot with Streaming

```bash
# Start the application with streaming enabled
streamlit run app.py --server.port 5000
```

Visit `http://localhost:5000` to access the chatbot with streaming responses.

## Usage

### Method 1: Upload via Web Interface
1. Use the file uploader in the sidebar
2. Select your document (PDF, DOCX, or TXT)
3. Wait for processing to complete
4. Start asking questions

### Method 2: Manual File Addition
1. Add your documents to the `data/` folder
2. The Document Library will show available files
3. Click "Process" for unprocessed documents
4. Click document name to load processed documents

## Project Structure

```
ai-document-chatbot/
├── app.py                 # Main Streamlit application
├── src/
│   ├── document_processor.py  # Document parsing and chunking
│   ├── vector_store.py        # FAISS vector database
│   ├── rag_pipeline.py        # RAG orchestration
│   ├── groq_handler.py        # Groq API integration
│   ├── llm_handler.py         # Local LLM fallback
│   └── utils.py               # Utility functions
├── data/                  # Document storage
├── chunks/                # Processed document chunks
├── vectordb/              # Vector indices and metadata
├── notebooks/             # Development notebooks
├── requirements.txt       # Python dependencies
└── README.md
```

## Model and Embedding Choices

### Embedding Model Selection
**SentenceTransformers (all-MiniLM-L6-v2)**
- **Rationale**: Optimized for semantic similarity tasks with good performance/size balance
- **Dimensions**: 384-dimensional vectors (efficient storage and fast retrieval)
- **Performance**: Excellent for document similarity and question-answering tasks
- **Fallback**: TF-IDF vectorization when SentenceTransformers unavailable

### Language Model Integration
**Primary: Groq LLaMA3-8B**
- **Rationale**: Fast inference, high-quality responses, cost-effective
- **Streaming**: Native streaming support for real-time response generation
- **Context Window**: 8192 tokens for comprehensive context processing
- **Response Format**: Generates 250-300 word professional responses with bullet points

**Fallback: OpenAI GPT-4o**
- **Rationale**: High-quality responses when Groq unavailable
- **Streaming**: Full streaming support with chunk-by-chunk delivery
- **Context Window**: Large context for complex document analysis

### Vector Database Choice
**FAISS (Facebook AI Similarity Search)**
- **Rationale**: Efficient similarity search with excellent performance
- **Index Type**: Flat index for exact similarity search
- **Persistence**: Automatic save/load of indices and metadata
- **Scalability**: Handles large document collections efficiently

## System Requirements

- Python 3.8+
- 4GB+ RAM (for document processing and vector operations)
- Internet connection (for API calls and model downloads)

## RAG Pipeline Configuration

### Preprocessing Parameters
```python
# Document Chunking
CHUNK_SIZE_RANGE = (100, 300)  # Words per chunk
CHUNK_OVERLAP = 50             # Overlap between chunks

# Vector Search
RETRIEVAL_K = 5                # Top chunks to retrieve
SIMILARITY_THRESHOLD = 0.1     # Minimum similarity score

# Response Generation
MAX_TOKENS = 450               # Maximum response length
TEMPERATURE = 0.7              # Response creativity balance
```

### Streaming Response Implementation
The chatbot implements real-time streaming responses:

1. **Generator Pattern**: Response generation uses Python generators
2. **Chunk Collection**: Streaming chunks collected into complete responses
3. **Real-time Display**: Streamlit displays responses as they generate
4. **Error Handling**: Graceful fallback for generator object issues

## API Integration

### Groq API (Recommended)
- **Model**: LLaMA3-8B
- **Speed**: Ultra-fast inference (~100ms response time)
- **Quality**: High-quality responses with structured formatting
- **Streaming**: Native streaming support for real-time display
- **Setup**: Get API key from [Groq Console](https://console.groq.com)

### OpenAI API (Alternative)
- **Model**: GPT-4o (latest model, released May 2024)
- **Quality**: Premium response quality with advanced reasoning
- **Streaming**: Full streaming support with delta responses
- **Context**: Large context window for complex documents
- **Setup**: Get API key from [OpenAI Platform](https://platform.openai.com)

## Configuration and Customization

### Automatic Configuration
The system automatically handles:
- Document processing and chunking
- Vector embedding generation
- Similarity search optimization
- Response generation and formatting
- Streaming response collection

### Manual Configuration Options
```python
# Update RAG pipeline parameters
pipeline.update_retrieval_params(
    k=10,                    # Retrieve more chunks
    threshold=0.2            # Higher similarity threshold
)

# Adjust LLM generation parameters
llm_handler.update_generation_params(
    max_length=600,          # Longer responses
    temperature=0.5          # More focused responses
)
```

### Environment Variables
```bash
# Core Configuration
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key

# Optional Configuration
CHUNK_SIZE_MIN=100
CHUNK_SIZE_MAX=300
VECTOR_DIMENSIONS=384
MAX_RETRIEVAL_CHUNKS=5
```

## Detailed Pipeline Steps

### Step 1: Document Preprocessing
```bash
# The preprocessing pipeline handles:
# 1. Text extraction from PDF/DOCX/TXT
# 2. Text cleaning and normalization  
# 3. Semantic chunking with context preservation
# 4. Metadata extraction and storage

# Files created:
# - chunks/document_name_chunks.json (processed text chunks)
# - data/document_name.pdf (original document)
```

### Step 2: Embedding Generation
```bash
# Vector embedding process:
# 1. Load SentenceTransformers model (all-MiniLM-L6-v2)
# 2. Generate 384-dimensional vectors for each chunk
# 3. Create FAISS index for similarity search
# 4. Save index and metadata

# Files created:
# - vectordb/index.faiss (vector index)
# - vectordb/metadata.json (chunk metadata)
# - vectordb/config.json (index configuration)
```

### Step 3: RAG Pipeline Execution
```bash
# Query processing flow:
# 1. User question → embedding vector
# 2. FAISS similarity search → top-K chunks
# 3. Context creation from relevant chunks
# 4. Prompt engineering for LLM
# 5. Streaming response generation
# 6. Source citation compilation
```

### Step 4: Streaming Response Architecture
```python
# Streaming implementation:
def generate_streaming_response(prompt):
    """
    1. LLM generates response chunks
    2. Chunks collected in real-time
    3. Streamlit displays progressive text
    4. Complete response stored in chat history
    """
    for chunk in llm_response:
        yield chunk  # Real-time streaming
        full_response += chunk  # Complete collection
```

## Troubleshooting

### Common Issues

1. **Generator Object Error**: Fixed in latest version - responses now properly collected
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **API Key Issues**: Ensure environment variables are set correctly
4. **Memory Issues**: Use smaller documents or increase system RAM
5. **Streaming Issues**: Check network connection and API rate limits

### Performance Optimization

- **Document Size**: 10,000+ words work best for meaningful chunking
- **PDF Quality**: High-quality PDFs extract better text
- **Processing Time**: Scales with document size and complexity
- **Memory Usage**: ~100MB per 1000 document chunks
- **API Latency**: Groq typically faster than OpenAI for streaming

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
streamlit run app.py --server.port 5000

# Check pipeline validation
python -c "
from src.rag_pipeline import RAGPipeline
# ... initialize pipeline ...
print(pipeline.validate_setup())
"
```

For issues and questions:
- Check the troubleshooting section
- Review the logs in the Streamlit interface
- Ensure all dependencies are properly installed