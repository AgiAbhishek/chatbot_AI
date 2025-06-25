import json
import os
from typing import Generator, List, Optional

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

class OpenAIHandler:
    """Handles OpenAI API integration for better response generation"""
    
    def __init__(self):
        self.openai_available = OPENAI_AVAILABLE
        self.client = None
        
        if self.openai_available:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                print("OpenAI API initialized successfully")
            else:
                print("OpenAI API key not found, will use fallback")
                self.openai_available = False
    
    def generate_response(self, prompt: str) -> Generator[str, None, None]:
        """Generate streaming response using OpenAI"""
        if not self.openai_available or not self.client:
            yield from self._fallback_response(prompt)
            return
        
        try:
            # Create a more conversational prompt
            system_prompt = """You are a helpful AI assistant that answers questions based on document context. 

Guidelines:
- Give short, clear, conversational responses (2-3 sentences max)
- Answer directly without mentioning "based on the context" repeatedly
- Use simple language, avoid jargon
- If information is missing, briefly state what you found instead
- Be helpful and natural like ChatGPT"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                max_tokens=200,  # Shorter, more focused responses
                temperature=0.7
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"OpenAI API error: {e}")
            yield from self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> Generator[str, None, None]:
        """Fallback response when OpenAI is not available"""
        try:
            # Extract question and context from prompt
            if "QUESTION:" in prompt and "CONTEXT:" in prompt:
                context_part = prompt.split("CONTEXT:")[1].split("QUESTION:")[0].strip()
                question_part = prompt.split("QUESTION:")[1].split("ANSWER:")[0].strip()
                
                # Create a simple but readable response
                response = self._create_readable_response(question_part, context_part)
            else:
                response = "I understand your question, but I need document context to provide an accurate answer. Please ensure a document is properly loaded."
            
            # Stream the response word by word
            words = response.split()
            for word in words:
                yield word + " "
                
        except Exception as e:
            yield f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def _create_readable_response(self, question: str, context: str) -> str:
        """Create a more readable response from context"""
        if not context.strip():
            return "I don't have enough information in the document to answer your question."
        
        # Simple keyword matching and sentence extraction
        question_words = set(word.lower() for word in question.split() if len(word) > 3)
        
        # Split context into sentences and find relevant ones
        sentences = []
        for line in context.split('\n'):
            if line.strip() and '[Source' not in line:
                sentences.extend([s.strip() for s in line.split('.') if s.strip()])
        
        # Find sentences with question keywords
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in question_words):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # Create a readable response
            response = "Based on the document, " + ". ".join(relevant_sentences[:2])
            if len(response) > 200:
                response = response[:200] + "..."
            return response
        else:
            # Provide the most relevant content
            first_meaningful_content = next((s for s in sentences if len(s) > 50), "")
            if first_meaningful_content:
                return f"The document contains information about this topic: {first_meaningful_content[:150]}..."
            return "I found information in the document, but it may not directly address your specific question. Please try rephrasing your question."
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return self.openai_available and self.client is not None