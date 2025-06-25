import os
import json
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np

# Embeddings - using scikit-learn as fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Simple TF-IDF based embeddings as fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vector store
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

class VectorStore:
    """Handles vector embeddings and similarity search using FAISS"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_db_path: str = "vectordb"):
        self.model_name = model_name
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(exist_ok=True)
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.use_sentence_transformers = True
        else:
            # Fallback to TF-IDF
            self.embedding_model = TfidfVectorizer(max_features=384, stop_words='english')
            self.embedding_dim = 384
            self.use_sentence_transformers = False
            print("Using TF-IDF embeddings as fallback (sentence-transformers not available)")
        
        # Initialize storage
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.fitted_vectorizer = False
        
        # Try to load existing index
        self._load_index()
    
    def _load_index(self):
        """Load existing index if available"""
        index_path = self.vector_db_path / "index.pkl"
        metadata_path = self.vector_db_path / "metadata.json"
        chunks_path = self.vector_db_path / "chunks.pkl"
        
        if all(path.exists() for path in [index_path, metadata_path, chunks_path]):
            try:
                # Load index
                with open(index_path, 'rb') as f:
                    self.index = pickle.load(f)
                
                # Load metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.chunk_metadata = json.load(f)
                
                # Load chunks
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                
                self.fitted_vectorizer = True
                print(f"Loaded existing index with {len(self.chunks)} chunks")
                
            except Exception as e:
                print(f"Error loading existing index: {e}")
                self._initialize_new_index()
        else:
            self._initialize_new_index()
    
    def _initialize_new_index(self):
        """Initialize a new index"""
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.fitted_vectorizer = False
    
    def add_documents(self, chunks: List[Dict[str, any]]) -> bool:
        """Add document chunks to the vector store"""
        try:
            if not chunks:
                raise ValueError("No chunks provided")
            
            # Clear existing data
            self._initialize_new_index()
            
            # Extract text for embedding
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings
            print(f"Generating embeddings for {len(texts)} chunks...")
            
            if self.use_sentence_transformers:
                embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
                self.index = embeddings.astype('float32')
            else:
                # Use TF-IDF
                tfidf_matrix = self.embedding_model.fit_transform(texts)
                self.index = tfidf_matrix.toarray().astype('float32')
                self.fitted_vectorizer = True
            
            # Store chunks and metadata
            self.chunks = chunks
            self.chunk_metadata = [
                {
                    'chunk_id': chunk['chunk_id'],
                    'word_count': chunk['word_count'],
                    'text_preview': chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                }
                for chunk in chunks
            ]
            
            # Save to disk
            self._save_index()
            
            print(f"Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, any]]:
        """Search for similar chunks"""
        try:
            if self.index is None or len(self.chunks) == 0:
                return []
            
            if self.use_sentence_transformers:
                # Generate query embedding
                query_embedding = self.embedding_model.encode([query])
                
                # Calculate cosine similarity
                similarities = cosine_similarity(query_embedding, self.index)[0]
            else:
                # Use TF-IDF
                if not self.fitted_vectorizer:
                    return []
                
                query_tfidf = self.embedding_model.transform([query])
                similarities = cosine_similarity(query_tfidf, self.index)[0]
            
            # Get top k similar chunks
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Prepare results
            results = []
            for i, idx in enumerate(top_indices):
                score = float(similarities[idx])
                print(f"Chunk {idx} similarity score: {score:.4f}")
                if score >= 0:  # Include all non-negative similarities
                    results.append({
                        'content': self.chunks[idx]['text'],
                        'score': score,
                        'chunk_id': self.chunks[idx]['chunk_id'],
                        'word_count': self.chunks[idx]['word_count'],
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def _save_index(self):
        """Save index and metadata to disk"""
        try:
            # Save index
            index_path = self.vector_db_path / "index.pkl"
            with open(index_path, 'wb') as f:
                pickle.dump(self.index, f)
            
            # Save metadata
            metadata_path = self.vector_db_path / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunk_metadata, f, indent=2, ensure_ascii=False)
            
            # Save chunks
            chunks_path = self.vector_db_path / "chunks.pkl"
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get vector store statistics"""
        return {
            'total_chunks': len(self.chunks),
            'embedding_model': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'index_size': len(self.chunks) if self.index is not None else 0,
            'total_words': sum(chunk.get('word_count', 0) for chunk in self.chunks)
        }
    
    def reset(self):
        """Reset the vector store"""
        self._initialize_new_index()
        
        # Clean up saved files
        for file_name in ['index.pkl', 'metadata.json', 'chunks.pkl']:
            file_path = self.vector_db_path / file_name
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Error deleting {file_name}: {e}")
    
    def is_empty(self) -> bool:
        """Check if vector store is empty"""
        return len(self.chunks) == 0
