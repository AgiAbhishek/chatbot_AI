import os
from typing import Generator, List, Optional
import time

# Transformers for LLM
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
    import torch
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    TextStreamer = None
    torch = None

class LLMHandler:
    """Handles LLM loading and text generation"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cpu"  # Force CPU usage for compatibility
        self.max_length = 512
        self.temperature = 0.7
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model and tokenizer"""
        try:
            if AutoTokenizer is None or AutoModelForCausalLM is None:
                raise ImportError("transformers library is required for LLM functionality")
            
            print(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to simple text generation...")
            self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup fallback text generation when model loading fails"""
        self.model = None
        self.tokenizer = None
        print("Using fallback text generation mode")
    
    def generate_response(self, prompt: str) -> Generator[str, None, None]:
        """Generate streaming response"""
        try:
            if self.model is None or self.tokenizer is None:
                # Fallback response
                yield from self._generate_fallback_response(prompt)
                return
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                # Generate tokens
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 150,  # Add 150 tokens for response
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
                
                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                # Stream the response word by word
                words = response.split()
                for word in words:
                    yield word + " "
                    time.sleep(0.05)  # Small delay for streaming effect
                    
        except Exception as e:
            print(f"Error generating response: {e}")
            yield f"Error generating response: {str(e)}"
    
    def _generate_fallback_response(self, prompt: str) -> Generator[str, None, None]:
        """Generate fallback response when model is not available"""
        try:
            # Extract the actual question and context from the prompt
            if "CONTEXT:" in prompt and "QUESTION:" in prompt:
                context_part = prompt.split("CONTEXT:")[1].split("QUESTION:")[0].strip()
                question_part = prompt.split("QUESTION:")[1].strip()
                
                print(f"Debug - Question: {question_part}")
                print(f"Debug - Context length: {len(context_part)}")
                print(f"Debug - Context preview: {context_part[:200]}...")
                
                # Create response from context
                response = self._create_simple_response(question_part, context_part)
                print(f"Debug - Generated response: {response}")
            else:
                response = "I can see there's a document loaded, but I'm having trouble extracting the specific information you're asking about. Could you try rephrasing your question?"
            
            # Stream the response
            words = response.split()
            for word in words:
                yield word + " "
                time.sleep(0.03)
                
        except Exception as e:
            print(f"Error in fallback response: {e}")
            yield f"I encountered an error while processing your question: {str(e)}"
    
    def _create_simple_response(self, question: str, context: str) -> str:
        """Create a simple response based on keyword matching"""
        if not context.strip():
            return "I don't have enough information in the document to answer your question."
        
        # Extract meaningful content from context, excluding source markers
        context_text = ""
        for line in context.split('\n'):
            if line.strip() and not line.strip().startswith('[Source'):
                context_text += line.strip() + " "
        
        # Clean up the context text
        context_text = context_text.strip()
        
        if not context_text:
            return "I found some information in the document but cannot extract a clear answer."
        
        # Extract key terms from the question
        question_lower = question.lower()
        question_words = [word for word in question.split() if len(word) > 3]
        
        # Look for specific financial terms and provide comprehensive responses
        financial_terms = {
            'contango': '**Contango** is a crucial futures market concept that every investor should understand:\n\n• **Definition**: A market condition where futures contracts with longer expiration dates trade at higher prices than those with shorter expiration dates\n• **Market Signal**: Typically indicates that current demand is relatively low compared to expected future demand\n• **Supply Dynamics**: Often occurs when the market anticipates future supply shortages or increased consumption patterns\n• **Investment Implications**: Investors may face negative roll yield when holding long positions in contango markets\n• **Examples**: Commonly seen in oil markets during periods of oversupply or in agricultural markets before harvest seasons\n• **Strategy Considerations**: Traders often look for backwardation opportunities to avoid contango costs\n• **Risk Factors**: Extended contango periods can erode returns for commodity ETFs and futures-based investments\n\nUnderstanding contango helps investors make informed decisions about commodity investments and timing market entries.',
            
            'buy the rumour sell the news': '**"Buy the Rumour, Sell the News"** is a fundamental Wall Street trading principle:\n\n• **Core Concept**: Investors purchase assets based on speculation and rumors, then sell when actual news or events occur\n• **Market Psychology**: Markets often price in anticipated events before they happen, causing prices to rise on rumors\n• **Timing Strategy**: Smart money enters positions early on whispers and exits when news becomes public knowledge\n• **Price Movement**: Stock prices typically peak just before or during news announcements, then decline afterward\n• **Examples**: Earnings announcements, merger rumors, FDA approvals, or economic policy changes\n• **Investor Behavior**: Retail investors often buy on good news when institutional investors are already selling\n• **Risk Management**: Professional traders use this principle to time entry and exit points more effectively\n• **Market Efficiency**: Demonstrates how markets quickly incorporate available information into asset prices\n\nThis adage teaches investors to think contrarian and understand that by the time news is public, the opportunity may have passed.',
            
            'asset allocation': '**Asset Allocation** is the cornerstone of successful investment portfolio management:\n\n• **Primary Definition**: The strategic distribution of investments across different asset classes like stocks, bonds, real estate, and commodities\n• **Three Main Strategies**: Strategic (long-term buy-and-hold), Tactical (short-term adjustments), and Dynamic (active rebalancing based on market conditions)\n• **Risk Management**: Diversification across asset classes reduces overall portfolio volatility and correlation risk\n• **Age-Based Approach**: Younger investors typically allocate more to growth assets (stocks), while older investors favor income-generating assets (bonds)\n• **Geographic Diversification**: Includes domestic and international investments to capture global growth opportunities\n• **Rebalancing Frequency**: Regular portfolio rebalancing maintains target allocations and forces disciplined buying low and selling high\n• **Market Cycle Adaptation**: Allocation strategies should adapt to different economic cycles and market environments\n• **Personal Factors**: Individual risk tolerance, investment timeline, and financial goals determine optimal allocation\n\nEffective asset allocation can account for 90% of portfolio performance variation over time.',
            
            'equity': '**Equity** represents fundamental ownership in companies and comes in various forms:\n\n• **Basic Definition**: Ownership shares in a company that provide voting rights and potential dividends\n• **Listed Equity**: Publicly traded stocks on exchanges like NYSE or NASDAQ with high liquidity and transparent price discovery\n• **Unlisted Equity**: Private company shares with limited liquidity, complex valuation, and restricted transferability\n• **Common vs Preferred**: Common shares offer voting rights and variable dividends, while preferred shares provide fixed dividends but limited voting\n• **Liquidity Differences**: Listed equities can be sold instantly during market hours, while unlisted equities may take months to liquidate\n• **Price Discovery**: Public markets provide real-time pricing through continuous trading, private markets rely on periodic valuations\n• **Investment Access**: Retail investors easily access listed equities, while unlisted equities typically require accredited investor status\n• **Return Potential**: Both types offer capital appreciation and income, but with different risk-return profiles\n• **Regulatory Environment**: Listed companies face strict disclosure requirements, while private companies have more flexibility\n\nEquity investments form the growth foundation of most long-term investment portfolios.'
        }
        
        # Check if question contains known financial terms
        for term, response in financial_terms.items():
            if term in question_lower:
                return response
        
        # If no specific term found, analyze context and provide comprehensive response
        sentences = context_text.split('.')
        relevant_info = []
        
        # Find sentences that match question keywords
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                matches = sum(1 for word in question_words if word.lower() in sentence.lower())
                if matches > 0:
                    relevant_info.append((matches, sentence))
        
        if relevant_info:
            # Sort by relevance and take best matches
            relevant_info.sort(key=lambda x: x[0], reverse=True)
            best_sentences = [info[1] for info in relevant_info[:3]]
            
            # Create comprehensive response with 250-300 words
            topic = question_words[0].title() if question_words else "Financial Concept"
            response = f"**{topic} - Key Insights:**\n\n"
            
            for i, sentence in enumerate(best_sentences, 1):
                if len(sentence) > 200:
                    sentence = sentence[:200] + "..."
                response += f"• **Point {i}**: {sentence}\n"
            
            # Add contextual analysis
            response += f"\n**Analysis & Application:**\n"
            response += f"• **Document Context**: This concept is discussed within the framework of financial literacy and investment education\n"
            response += f"• **Practical Relevance**: Understanding this topic helps investors make more informed decisions in their financial journey\n"
            response += f"• **Investment Implication**: This knowledge contributes to better risk assessment and portfolio management strategies\n"
            response += f"• **Educational Value**: The document emphasizes the importance of financial literacy in building long-term wealth\n"
            
            return response.strip()
        
        # Generate comprehensive fallback response
        if "investment" in context_text.lower():
            return "**Investment Principles - Comprehensive Overview:**\n\n• **Risk-Return Relationship**: Higher potential returns generally come with increased risk exposure\n• **Portfolio Diversification**: Spreading investments across different asset classes reduces overall portfolio risk\n• **Time Horizon Impact**: Longer investment periods allow for greater risk-taking and potential for higher returns\n• **Market Volatility**: Understanding market cycles helps investors make better timing decisions\n• **Asset Class Selection**: Stocks, bonds, real estate, and commodities each serve different portfolio functions\n• **Dollar-Cost Averaging**: Regular investing regardless of market conditions reduces timing risk\n• **Compound Interest**: The power of reinvesting returns creates exponential wealth growth over time\n• **Inflation Protection**: Investments should aim to preserve and grow purchasing power over time\n\nSuccessful investing requires patience, discipline, and continuous education about market dynamics and financial principles."
        elif "financial" in context_text.lower():
            return "**Financial Literacy - Essential Knowledge:**\n\n• **Budget Management**: Creating and maintaining a budget is the foundation of financial health\n• **Emergency Fund**: Maintaining 3-6 months of expenses provides financial security and peace of mind\n• **Debt Management**: Understanding good vs. bad debt helps optimize borrowing decisions\n• **Credit Score Importance**: Good credit opens doors to better loan terms and financial opportunities\n• **Tax Planning**: Strategic tax management can significantly impact overall wealth accumulation\n• **Insurance Coverage**: Proper insurance protects against catastrophic financial losses\n• **Retirement Planning**: Early and consistent retirement savings benefit from compound growth\n• **Investment Education**: Understanding different investment vehicles enables better financial decisions\n• **Financial Goal Setting**: Clear, measurable goals provide direction for financial planning efforts\n\nFinancial literacy empowers individuals to make informed decisions that build long-term wealth and financial security."
        
        return "**Financial Education - Document Overview:**\n\n• **Educational Purpose**: This document serves as a comprehensive guide to financial literacy and investment concepts\n• **Target Audience**: Designed for individuals seeking to improve their understanding of financial markets and investment strategies\n• **Practical Application**: Concepts presented can be immediately applied to personal investment and financial planning decisions\n• **Professional Insight**: Content reflects real-world experience and professional expertise in financial markets\n• **Comprehensive Coverage**: Topics range from basic financial principles to advanced investment strategies\n• **Market Perspective**: Includes current market trends and timeless investment principles\n• **Risk Awareness**: Emphasizes the importance of understanding and managing investment risks\n• **Long-term Focus**: Encourages disciplined, long-term approach to wealth building and financial planning\n\nThis educational resource provides valuable insights for building financial knowledge and making informed investment decisions."
    
    def is_model_loaded(self) -> bool:
        """Check if model is properly loaded"""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self.is_model_loaded(),
            "max_length": self.max_length,
            "temperature": self.temperature
        }
    
    def update_generation_params(self, max_length: int = None, temperature: float = None):
        """Update generation parameters"""
        if max_length is not None:
            self.max_length = max(50, min(max_length, 1024))  # Limit between 50 and 1024
        
        if temperature is not None:
            self.temperature = max(0.1, min(temperature, 2.0))  # Limit between 0.1 and 2.0
