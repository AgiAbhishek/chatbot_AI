from typing import Dict, Any, List, Optional
import os
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.llm_handler import LLMHandler

# Try to import Groq handler
try:
    from src.groq_handler import GroqHandler
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

class RAGPipeline:
    """Main RAG pipeline orchestrating document processing, retrieval, and generation"""
    
    def __init__(self, document_processor: DocumentProcessor, vector_store: VectorStore, llm_handler: LLMHandler):
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.llm_handler = llm_handler
        
        # Initialize Groq handler if available
        if GROQ_AVAILABLE:
            self.groq_handler = GroqHandler()
        else:
            self.groq_handler = None
        
        # Configuration
        self.retrieval_k = 5
        self.similarity_threshold = 0.1
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a document and add it to the vector store"""
        try:
            # Process document into chunks
            chunks = self.document_processor.process_document(file_path)
            
            # Add chunks to vector store
            success = self.vector_store.add_documents(chunks)
            
            if success:
                print(f"Successfully processed document with {len(chunks)} chunks")
                return chunks
            else:
                raise Exception("Failed to add chunks to vector store")
                
        except Exception as e:
            print(f"Error processing document: {e}")
            raise e
    
    def retrieve_relevant_chunks(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query"""
        try:
            # Search for relevant chunks
            chunks = self.vector_store.search(query, k=self.retrieval_k)
            
            # Filter by similarity threshold
            relevant_chunks = [
                chunk for chunk in chunks 
                if chunk.get('score', 0) >= self.similarity_threshold
            ]
            
            print(f"Found {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def create_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Create context string from retrieved chunks"""
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Source {i}]\n{chunk['content']}\n")
        
        return "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """Create the prompt for the LLM"""
        prompt_template = """Provide a comprehensive 250-300 word response with detailed bullet points and professional formatting.

CONTEXT:
{context}

QUESTION: {query}

Provide a detailed, professional response with:
- Bold headings for key concepts
- Detailed bullet points with explanations
- Practical applications and implications
- 250-300 words total
- Educational value and thorough coverage

Response:"""
        
        return prompt_template.format(context=context, query=query)
    
    def get_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get complete RAG response for a query"""
        try:
            # Check if vector store has data
            if self.vector_store.is_empty():
                return {
                    "response": "Please upload and process a document first.",
                    "sources": []
                }
            
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query)
            
            # Debug: show what chunks were found
            print(f"Query: {query[:50]}...")
            print(f"Found {len(relevant_chunks)} relevant chunks")
            
            if not relevant_chunks:
                # Try without similarity threshold as fallback
                print("No chunks found with threshold, trying without threshold...")
                all_chunks = self.vector_store.search(query, k=self.retrieval_k)
                if all_chunks:
                    relevant_chunks = all_chunks[:3]  # Take top 3 regardless of score
                    print(f"Using {len(relevant_chunks)} chunks without threshold")
                
            if not relevant_chunks:
                return {
                    "response": "I couldn't find relevant information in the document to answer your question. Please try rephrasing your question or ask about different topics covered in the document.",
                    "sources": []
                }
            
            # Create context
            context = self.create_context(relevant_chunks)
            
            # Generate response using Groq if available, otherwise fallback
            if self.groq_handler and self.groq_handler.is_available():
                prompt = self.create_prompt(query, context)
                response_generator = self.groq_handler.generate_response(prompt)
            else:
                prompt = self.create_prompt(query, context)
                response_generator = self.llm_handler.generate_response(prompt)
            
            # FIXED: Collect the streaming response into a complete string
            response_text = ""
            try:
                for chunk in response_generator:
                    if chunk:
                        response_text += str(chunk)
            except Exception as e:
                print(f"Error collecting response: {e}")
                response_text = "Sorry, I encountered an error generating the response."
            
            return {
                "query": query,
                "response": response_text,  # Return complete text instead of generator
                "sources": relevant_chunks,
                "context": context
            }
            
        except Exception as e:
            print(f"Error in RAG pipeline: {e}")
            return {
                "response": f"An error occurred: {str(e)}",
                "sources": []
            }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG pipeline"""
        try:
            vector_stats = self.vector_store.get_stats()
            llm_info = self.llm_handler.get_model_info() if hasattr(self.llm_handler, 'get_model_info') else {}
            
            return {
                "total_chunks": vector_stats.get('total_chunks', 0),
                "vector_dimensions": vector_stats.get('vector_dimensions', 0),
                "model_name": vector_stats.get('model_name', 'Unknown'),
                "llm_model": llm_info.get('model_name', 'Fallback'),
                "groq_available": self.groq_handler.is_available() if self.groq_handler else False
            }
        except Exception as e:
            print(f"Error getting pipeline stats: {e}")
            return {}
    
    def reset(self):
        """Reset the entire pipeline"""
        try:
            self.vector_store.reset()
            print("Pipeline reset successfully")
        except Exception as e:
            print(f"Error resetting pipeline: {e}")
    
    def update_retrieval_params(self, k: int = None, threshold: float = None):
        """Update retrieval parameters"""
        if k is not None:
            self.retrieval_k = k
        if threshold is not None:
            self.similarity_threshold = threshold
        
        print(f"Updated retrieval params: k={self.retrieval_k}, threshold={self.similarity_threshold}")
    
    def validate_setup(self) -> Dict[str, bool]:
        """Validate that all components are properly set up"""
        return {
            "document_processor": self.document_processor is not None,
            "vector_store": self.vector_store is not None,
            "llm_handler": self.llm_handler is not None,
            "groq_handler": self.groq_handler is not None and self.groq_handler.is_available(),
            "vector_store_ready": not self.vector_store.is_empty()
        }