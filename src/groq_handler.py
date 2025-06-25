import json
import os
from typing import Generator, List, Optional

try:
    from openai import OpenAI
    GROQ_AVAILABLE = True
except ImportError:
    OpenAI = None
    GROQ_AVAILABLE = False

class GroqHandler:
    """Handles Groq API integration for better response generation"""
    
    def __init__(self):
        self.groq_available = GROQ_AVAILABLE
        self.client = None
        
        if self.groq_available:
            try:
                # Use the provided Groq API key
                api_key = "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"
                self.client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=api_key
                )
                print("Groq API initialized successfully with LLaMA3-8B model")
            except Exception as e:
                print(f"Failed to initialize Groq: {e}")
                self.groq_available = False
        else:
            print("OpenAI library not available - using fallback responses")
    
    def generate_response(self, prompt: str) -> Generator[str, None, None]:
        """Generate streaming response using Groq"""
        if not self.groq_available or not self.client:
            yield from self._fallback_response(prompt)
            return
        
        try:
            # Create a conversational system prompt
            system_prompt = """You are a professional financial AI assistant that provides comprehensive responses with expandable content.

PART 1 (Summary - 80-100 words):
- Brief overview with key concept in bold
- 2-3 main bullet points
- Concise explanation

PART 2 (Detailed Content - 200-250 words):  
- Comprehensive explanation with detailed bullet points
- Practical applications and implications
- Educational value with thorough coverage
- Bold headings for key concepts
- Multiple detailed bullet points with explanations

Example format:
**Modern Portfolio Theory (MPT)**

Overview: Brief explanation here...

• Key point 1
• Key point 2  
• Key point 3

---READ MORE---

**Detailed Analysis**

• **Component 1**: Detailed explanation...
• **Component 2**: Detailed explanation...
• **Applications**: Practical uses...
• **Implications**: What this means for investors..."""

            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                max_tokens=450,  # Expandable responses with Read More
                temperature=0.7
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"Groq API error: {e}")
            yield from self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> Generator[str, None, None]:
        """Fallback response when Groq is not available"""
        try:
            # Extract question and context from prompt
            if "QUESTION:" in prompt and "CONTEXT:" in prompt:
                context_part = prompt.split("CONTEXT:")[1].split("QUESTION:")[0].strip()
                question_part = prompt.split("QUESTION:")[1].strip()
                
                # Create a simple but readable response
                response = self._create_readable_response(question_part, context_part)
            else:
                response = "I understand your question, but I need document context to provide an accurate answer."
            
            # Stream the response word by word
            words = response.split()
            for word in words:
                yield word + " "
                
        except Exception as e:
            yield f"I apologize, but I encountered an error: {str(e)}"
    
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
                sentences.extend([s.strip() for s in line.split('.') if s.strip() and len(s) > 20])
        
        # Find sentences with question keywords
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in question_words):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # Create a natural response
            response = relevant_sentences[0]
            if len(response) > 150:
                response = response[:150] + "..."
            return response
        else:
            # Provide the most relevant content
            if sentences:
                return sentences[0][:150] + "..." if len(sentences[0]) > 150 else sentences[0]
            return "The document contains related information, but I cannot find a direct answer to your specific question."
    
    def is_available(self) -> bool:
        """Check if Groq is available"""
        return self.groq_available and self.client is not None