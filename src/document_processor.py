import os
import re
import json
from typing import List, Dict, Optional
from pathlib import Path
import hashlib

# Document parsing libraries
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class DocumentProcessor:
    """Handles document parsing, cleaning, and chunking"""
    
    def __init__(self, chunk_size_range: tuple = (100, 300)):
        self.chunk_size_range = chunk_size_range
        self.stop_words = set(stopwords.words('english'))
        
    def load_document(self, file_path: str) -> str:
        """Load and extract text from various document formats"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.txt':
            return self._load_txt(file_path)
        elif file_extension == '.pdf':
            return self._load_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return self._load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_txt(self, file_path: Path) -> str:
        """Load text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load PDF file"""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF processing")
        
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
        
        return text
    
    def _load_docx(self, file_path: Path) -> str:
        """Load DOCX file"""
        if DocxDocument is None:
            raise ImportError("python-docx is required for DOCX processing")
        
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error reading DOCX: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\']+', '', text)
        
        # Remove headers/footers patterns (common patterns)
        patterns_to_remove = [
            r'Page \d+ of \d+',
            r'^\d+\s*$',  # Page numbers on separate lines
            r'(?i)copyright.*',
            r'(?i)confidential.*',
            r'(?i)proprietary.*',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # Remove extra whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_chunks(self, text: str) -> List[Dict[str, any]]:
        """Split text into semantic chunks of 100-300 words"""
        # First, split into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            sentence_word_count = len(sentence_words)
            
            # If adding this sentence would exceed max chunk size, finalize current chunk
            if (current_word_count + sentence_word_count > self.chunk_size_range[1] 
                and current_word_count >= self.chunk_size_range[0]):
                
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'word_count': current_word_count,
                        'chunk_id': len(chunks)
                    })
                
                current_chunk = [sentence]
                current_word_count = sentence_word_count
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'word_count': current_word_count,
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def process_document(self, file_path: str, save_chunks: bool = True) -> List[Dict[str, any]]:
        """Complete document processing pipeline"""
        try:
            # Load document
            raw_text = self.load_document(file_path)
            
            if len(raw_text.strip()) == 0:
                raise ValueError("Document appears to be empty")
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            # Create chunks
            chunks = self.create_chunks(cleaned_text)
            
            if len(chunks) == 0:
                raise ValueError("No chunks created from document")
            
            # Save chunks if requested
            if save_chunks:
                self._save_chunks(chunks, file_path)
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
    
    def _save_chunks(self, chunks: List[Dict[str, any]], original_file_path: str):
        """Save chunks to file"""
        chunks_dir = Path("chunks")
        chunks_dir.mkdir(exist_ok=True)
        
        # Create filename based on original file
        original_name = Path(original_file_path).stem
        chunks_file = chunks_dir / f"{original_name}_chunks.json"
        
        # Add metadata
        chunks_data = {
            'original_file': original_file_path,
            'total_chunks': len(chunks),
            'total_words': sum(chunk['word_count'] for chunk in chunks),
            'chunks': chunks
        }
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    def load_chunks(self, chunks_file_path: str) -> List[Dict[str, any]]:
        """Load previously saved chunks"""
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        return chunks_data['chunks']
    
    def get_document_stats(self, text: str) -> Dict[str, int]:
        """Get basic statistics about the document"""
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        return {
            'total_characters': len(text),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0
        }
