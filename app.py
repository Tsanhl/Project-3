"""
🤖 ChatGPT-Style AI Assistant with Real-Time Web Search
A conversational AI that answers any question using web search and fact-checking
"""

import streamlit as st
import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from crewai import Crew, Task, Agent
from crewai_tools import TavilySearchTool
import time
from datetime import datetime
from dotenv import load_dotenv

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

def initialize_llm(api_key: str, model: str = "llama3-8b-8192"):
    """Initialize the LLM with Groq - properly configured for CrewAI"""
    # IMPORTANT: Do NOT set OPENAI_API_KEY - CrewAI will use it to create OpenAI client
    # Instead, create ChatOpenAI instance with Groq's endpoint directly
    
    # Create LangChain ChatOpenAI instance configured for Groq
    # This MUST use Groq's endpoint, not OpenAI's default
    llm = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",  # Groq's OpenAI-compatible endpoint
        openai_api_key=api_key,  # Groq API key (stored in LLM instance, not env)
        model_name=model,  # Model name
        temperature=0.7,
        max_tokens=2000,
    )
    
    return llm

def setup_web_search(tavily_api_key: str):
    """Setup web search tool using CrewAI's TavilySearchTool"""
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    return TavilySearchTool()

def create_research_agent(llm: ChatOpenAI, web_search_tool: Any):
    """Create Research Agent for web search"""
    return Agent(
        role='Expert Researcher and Web Search Specialist',
        goal='Search the web for current, accurate information on any topic and provide comprehensive, fact-checked answers',
        backstory=(
            "You are an expert researcher with access to real-time web search. "
            "You excel at finding accurate, up-to-date information from the internet. "
            "You always verify facts from multiple sources and provide well-sourced answers. "
            "You can answer questions about ANY topic - current events, history, science, technology, culture, and more."
        ),
        verbose=False,  # Reduced verbosity for cleaner output
        allow_delegation=False,
        llm=llm,  # Pass the configured LLM directly
        tools=[web_search_tool]
    )

def create_answer_agent(llm: ChatOpenAI):
    """Create Answer Agent for ChatGPT-like responses"""
    return Agent(
        role='Helpful AI Assistant',
        goal='Provide clear, conversational, and accurate answers to user questions in a friendly, ChatGPT-like manner',
        backstory=(
            "You are a helpful AI assistant similar to ChatGPT. "
            "You provide clear, conversational answers to any question. "
            "You are friendly, knowledgeable, and always aim to be helpful. "
            "You format your answers in a natural, easy-to-read way with proper paragraphs and structure."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

def create_fact_checker_agent(llm: ChatOpenAI, web_search_tool: Any):
    """Create Fact Checker Agent"""
    return Agent(
        role='Fact Verification Specialist',
        goal='Verify all facts in the answer by cross-checking with web sources',
        backstory=(
            "You are an expert fact-checker. "
            "You verify every factual claim against current web sources. "
            "You ensure accuracy and provide corrections if needed."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm,
        tools=[web_search_tool]
    )

def get_ai_response(query: str, groq_api_key: str, tavily_api_key: str, model: str = "llama3-8b-8192"):
    """Get AI response using CrewAI agents"""
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
        
        # Initialize LLM and tools
        llm = initialize_llm(groq_api_key, model)
        web_search_tool = setup_web_search(tavily_api_key)
        
        # Create agents
        research_agent = create_research_agent(llm, web_search_tool)
        answer_agent = create_answer_agent(llm)
        fact_checker_agent = create_fact_checker_agent(llm, web_search_tool)
        
        # Create tasks with clear instructions
        research_task = Task(
            description=f"Search the web for comprehensive information about: {query}. "
                       f"Use your web search tool to find current, accurate information from multiple sources. "
                       f"Gather detailed information about this topic.",
            expected_output="Detailed research findings with sources from web search",
            agent=research_agent,
        )
        
        answer_task = Task(
            description=f"Based on the research about '{query}', provide a clear, conversational answer. "
                       f"Write naturally and helpfully, like ChatGPT. "
                       f"Make it well-structured with proper paragraphs. "
                       f"Be friendly and informative.",
            expected_output="A clear, conversational answer to the user's question",
            agent=answer_agent,
            context=[research_task]
        )
        
        fact_check_task = Task(
            description=f"Review the answer about '{query}' and verify all facts using web search. "
                       f"Cross-check any claims or statistics. "
                       f"Provide the final verified, accurate answer.",
            expected_output="A fact-checked, accurate, and verified answer",
            agent=fact_checker_agent,
            context=[answer_task]
        )
        
        # Create crew
        crew = Crew(
            agents=[research_agent, answer_agent, fact_checker_agent],
            tasks=[research_task, answer_task, fact_check_task],
            verbose=False,
            max_iter=3,
            max_rpm=10
        )
        
        # Execute with timeout handling
        try:
            # CRITICAL: Remove OPENAI_API_KEY from environment before execution
            # This prevents CrewAI from creating its own OpenAI client instead of using our Groq LLM
            original_openai_key = os.environ.pop("OPENAI_API_KEY", None)
            
            # Execute crew
            result = crew.kickoff(inputs={"query": query})
            
            # Restore original key if it existed (for other parts of the system)
            if original_openai_key:
                os.environ["OPENAI_API_KEY"] = original_openai_key
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                return "❌ Error: API key authentication failed.\n\n**Please verify:**\n1. Your Groq API key is correct (starts with 'gsk_')\n2. Your Tavily API key is correct (starts with 'tvly-')\n3. Keys are entered correctly in the sidebar\n4. Keys are active and not expired\n\n**Get free keys:**\n- Groq: https://console.groq.com/\n- Tavily: https://tavily.com/"
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return "⚠️ Error: Rate limit exceeded. Please wait a moment and try again."
            elif "openai" in error_msg.lower() and "api" in error_msg.lower():
                return "❌ Error: Configuration issue detected. The system is trying to use OpenAI instead of Groq.\n\n**Solution:**\n1. Make sure your Groq API key is entered in the sidebar\n2. The key should start with 'gsk_'\n3. Try refreshing the page and re-entering your keys"
            else:
                return f"❌ Error: {error_msg}\n\nPlease check your API keys and try again."
        
        # Extract result
        if hasattr(result, 'raw') and result.raw:
            return str(result.raw)
        elif hasattr(result, 'content') and result.content:
            return str(result.content)
        elif hasattr(result, 'tasks_output') and result.tasks_output:
            if isinstance(result.tasks_output, list) and len(result.tasks_output) > 0:
                return str(result.tasks_output[-1])
            else:
                return str(result.tasks_output)
        elif isinstance(result, dict):
            return result.get('output', result.get('result', str(result)))
        else:
            result_str = str(result)
            if len(result_str.strip()) < 10:
                return "⚠️ Response generated but was too short. Please try again or rephrase your question."
            return result_str
            
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
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
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
            with st.spinner("🔍 Searching the web and thinking..."):
                response = get_ai_response(prompt, groq_api_key, tavily_api_key, model)
            
            # Display response
            message_placeholder = st.empty()
            message_placeholder.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
