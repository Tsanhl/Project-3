"""
🤖 ChatGPT-Style AI Assistant with Real-Time Web Search
A conversational AI that answers any question using web search and fact-checking
Uses Groq and Tavily APIs directly for reliable performance
"""

import streamlit as st
import os
from typing import Dict, Any, List
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🤖 AI Assistant - ChatGPT Style",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like interface
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 0.75rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

if 'tavily_api_key' not in st.session_state:
    st.session_state.tavily_api_key = os.getenv("TAVILY_API_KEY", "")

def search_web(query: str, tavily_client: TavilyClient, max_results: int = 5) -> str:
    """Search the web using Tavily and return formatted results"""
    try:
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced"
        )
        
        # Format results for the LLM
        formatted_results = []
        if 'results' in response:
            for result in response['results']:
                formatted_results.append({
                    'title': result.get('title', 'No title'),
                    'url': result.get('url', 'No URL'),
                    'content': result.get('content', 'No content')
                })
        
        # Create a readable summary
        summary = "Web Search Results:\n\n"
        for i, result in enumerate(formatted_results, 1):
            summary += f"{i}. {result['title']}\n"
            summary += f"   URL: {result['url']}\n"
            summary += f"   Content: {result['content'][:500]}...\n\n"
        
        return summary if formatted_results else f"No results found for: {query}"
    except Exception as e:
        return f"Error searching web: {str(e)}"

def get_ai_response(query: str, groq_api_key: str, tavily_api_key: str, model: str = "llama-3.1-8b-instant"):
    """Get AI response using Groq and Tavily directly"""
    try:
        # Validate API keys first
        if not groq_api_key or groq_api_key.strip() == "":
            return "❌ Error: Groq API key is missing. Please enter your Groq API key in the sidebar."
        
        if not tavily_api_key or tavily_api_key.strip() == "":
            return "❌ Error: Tavily API key is missing. Please enter your Tavily API key in the sidebar."
        
        # Validate key format (basic check)
        if not groq_api_key.startswith('gsk_'):
            return "❌ Error: Invalid Groq API key format. Groq keys should start with 'gsk_'. Please check your key."
        
        if not tavily_api_key.startswith('tvly-'):
            return "❌ Error: Invalid Tavily API key format. Tavily keys should start with 'tvly-'. Please check your key."
        
        # Initialize clients
        try:
            groq_client = Groq(api_key=groq_api_key)
            tavily_client = TavilyClient(api_key=tavily_api_key)
        except Exception as e:
            return f"❌ Error initializing API clients: {str(e)}"
        
        # Step 1: Search the web for current information
        with st.spinner("🔍 Searching the web..."):
            web_results = search_web(query, tavily_client)
        
        # Step 2: Create prompt with web search results
        system_prompt = """You are a helpful AI assistant that answers questions using web search results.
You provide accurate, up-to-date information based on the search results provided.
Be conversational, clear, and cite sources when relevant.
If the search results don't contain enough information, say so honestly."""
        
        user_prompt = f"""Question: {query}

{web_results}

Please provide a clear, accurate answer to the question based on the web search results above.
If the search results contain relevant information, use it. If not, provide a general answer if you know it, or say you need more information.
Make your answer conversational and well-structured."""
        
        # Step 3: Get response from Groq
        with st.spinner("🤖 Generating answer..."):
            try:
                # Use Groq's chat completion API
                # Handle different model parameter requirements
                create_params = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "top_p": 1,
                }
                
                # Some models may need different parameters, try with default first
                try:
                    completion = groq_client.chat.completions.create(**create_params)
                except Exception as model_error:
                    # If model doesn't exist, try with llama-3.1-8b-instant as fallback
                    if "not found" in str(model_error).lower() or "decommissioned" in str(model_error).lower():
                        st.warning(f"⚠️ Model {model} not available, using llama-3.1-8b-instant instead")
                        create_params["model"] = "llama-3.1-8b-instant"
                        completion = groq_client.chat.completions.create(**create_params)
                    else:
                        raise model_error
                
                # Extract response
                response = completion.choices[0].message.content
                
                if not response or len(response.strip()) < 10:
                    return "⚠️ Response generated but was too short. Please try again or rephrase your question."
                
                return response
                
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                    return "❌ Error: API key authentication failed.\n\n**Please verify:**\n1. Your Groq API key is correct (starts with 'gsk_')\n2. Your Tavily API key is correct (starts with 'tvly-')\n3. Keys are entered correctly in the sidebar\n4. Keys are active and not expired\n\n**Get free keys:**\n- Groq: https://console.groq.com/\n- Tavily: https://tavily.com/"
                elif "rate limit" in error_msg.lower() or "429" in error_msg:
                    return "⚠️ Error: Rate limit exceeded. Please wait a moment and try again."
                else:
                    return f"❌ Error calling Groq API: {error_msg}\n\nPlease check your API keys and try again."
                    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        
        # Provide helpful error messages
        error_msg = str(e)
        if "401" in error_msg or "invalid_api_key" in error_msg.lower():
            return "❌ API Key Error: Please verify your API keys are correct and active.\n\n- Groq key should start with 'gsk_'\n- Tavily key should start with 'tvly-'\n\nGet free keys at:\n- https://console.groq.com/\n- https://tavily.com/"
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            return "⚠️ Connection Error: Please check your internet connection and try again."
        else:
            return f"❌ Error: {error_msg}\n\nPlease try again or check the sidebar for API key setup instructions."

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        <p style="font-size: 1.1rem;">Ask me anything! I search the web to give you accurate, up-to-date answers.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Keys Section
        st.markdown("### 🔑 API Keys")
        
        with st.expander("📖 How to Get FREE API Keys", expanded=False):
            st.markdown("""
            #### 🚀 Groq API Key (FREE)
            1. Visit **[Groq Console](https://console.groq.com/)**
            2. Sign up (free, no credit card)
            3. Go to **API Keys** section
            4. Click **"Create API Key"**
            5. Copy key (starts with `gsk_`)
            6. Paste below
            
            **Free:** 14,400 requests/day
            
            #### 🌐 Tavily API Key (FREE)
            1. Visit **[Tavily AI](https://tavily.com/)**
            2. Sign up (free)
            3. Go to Dashboard → **API Keys**
            4. Generate API key (starts with `tvly-`)
            5. Paste below
            
            **Free:** 1,000 searches/month
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("[🔗 Get Groq Key](https://console.groq.com/)")
            with col2:
                st.markdown("[🔗 Get Tavily Key](https://tavily.com/)")
        
        groq_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            value=st.session_state.groq_api_key,
            placeholder="gsk_...",
            help="Enter your Groq API key (should start with 'gsk_')"
        )
        st.session_state.groq_api_key = groq_api_key
        
        # Show validation status
        if groq_api_key and not groq_api_key.startswith('gsk_'):
            st.warning("⚠️ Groq keys should start with 'gsk_'")
        
        tavily_api_key = st.text_input(
            "🌐 Tavily API Key",
            type="password",
            value=st.session_state.tavily_api_key,
            placeholder="tvly-...",
            help="Enter your Tavily API key (should start with 'tvly-')"
        )
        st.session_state.tavily_api_key = tavily_api_key
        
        # Show validation status
        if tavily_api_key and not tavily_api_key.startswith('tvly-'):
            st.warning("⚠️ Tavily keys should start with 'tvly-'")
        
        st.divider()
        
        # Model selection
        model = st.selectbox(
            "🤖 Model",
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "openai/gpt-oss-20b"],
            index=0,
            help="Select the AI model (llama3-8b-8192 is fastest)"
        )
        
        st.divider()
        
        # Quick test button
        if st.button("🧪 Test API Keys", use_container_width=True):
            if not groq_api_key or not groq_api_key.startswith('gsk_'):
                st.error("❌ Invalid Groq API key format")
            elif not tavily_api_key or not tavily_api_key.startswith('tvly-'):
                st.error("❌ Invalid Tavily API key format")
            else:
                st.success("✅ API key formats look correct! Try asking a question.")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Info
        st.info("💡 **Tip:** I search the web for every question to give you accurate, up-to-date answers!")
    
    # Main chat interface
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Validate API keys
        if not groq_api_key or not tavily_api_key or groq_api_key.strip() == "" or tavily_api_key.strip() == "":
            st.error("❌ Please enter both API keys in the sidebar!")
            st.info("💡 Don't have API keys? Expand 'How to Get FREE API Keys' in the sidebar!")
            st.stop()
        
        # Validate key formats
        if not groq_api_key.startswith('gsk_'):
            st.error("❌ Invalid Groq API key! Keys should start with 'gsk_'\n\nGet your key at: https://console.groq.com/")
            st.stop()
        
        if not tavily_api_key.startswith('tvly-'):
            st.error("❌ Invalid Tavily API key! Keys should start with 'tvly-'\n\nGet your key at: https://tavily.com/")
            st.stop()
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response = get_ai_response(prompt, groq_api_key, tavily_api_key, model)
            
            # Display response
            message_placeholder = st.empty()
            message_placeholder.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
