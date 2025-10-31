"""
🤖 ChatGPT-Style AI Assistant with Real-Time Web Search
A conversational AI that answers any question using web search and fact-checking
"""

import streamlit as st
import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai import Crew, Task, Agent
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
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background: #f0f0f0;
        margin-left: 20%;
    }
    .assistant-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-right: 20%;
    }
    .message-content {
        flex: 1;
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
    """Initialize the LLM with Groq"""
    return ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=api_key,
        model_name=model,
        temperature=0.7,
        max_tokens=2000,
    )

def setup_web_search(tavily_api_key: str):
    """Setup web search tool"""
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    return TavilySearchResults(k=10, max_results=10)

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
        verbose=True,
        allow_delegation=False,
        llm=llm,
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
        verbose=True,
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
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[web_search_tool]
    )

def get_ai_response(query: str, groq_api_key: str, tavily_api_key: str):
    """Get AI response using CrewAI agents"""
    try:
        llm = initialize_llm(groq_api_key)
        web_search_tool = setup_web_search(tavily_api_key)
        
        # Create agents
        research_agent = create_research_agent(llm, web_search_tool)
        answer_agent = create_answer_agent(llm)
        fact_checker_agent = create_fact_checker_agent(llm, web_search_tool)
        
        # Create tasks
        research_task = Task(
            description=f"Search the web thoroughly for information about: {query}. "
                       f"Gather current, accurate information from multiple sources. "
                       f"Focus on finding the most relevant and up-to-date information.",
            expected_output="Comprehensive research findings with sources from web search",
            agent=research_agent,
        )
        
        answer_task = Task(
            description=f"Based on the research about '{query}', provide a clear, conversational answer. "
                       f"Answer the question naturally, like ChatGPT would. "
                       f"Make it friendly, well-structured, and easy to understand. "
                       f"Include relevant details and context.",
            expected_output="A clear, conversational answer to the user's question",
            agent=answer_agent,
            context=[research_task]
        )
        
        fact_check_task = Task(
            description=f"Verify the facts in the answer about '{query}'. "
                       f"Use web search to cross-check any claims, statistics, or facts. "
                       f"Ensure accuracy and provide the final verified answer.",
            expected_output="A fact-checked, accurate answer",
            agent=fact_checker_agent,
            context=[answer_task]
        )
        
        # Create crew
        crew = Crew(
            agents=[research_agent, answer_agent, fact_checker_agent],
            tasks=[research_task, answer_task, fact_check_task],
            verbose=False,
        )
        
        # Execute
        result = crew.kickoff(inputs={"query": query})
        
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
        else:
            return str(result)
            
    except Exception as e:
        return f"Error: {str(e)}"

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
            3. Create API key
            4. Copy and paste below
            
            **Free:** 14,400 requests/day
            
            #### 🌐 Tavily API Key (FREE)
            1. Visit **[Tavily AI](https://tavily.com/)**
            2. Sign up (free)
            3. Generate API key
            4. Copy and paste below
            
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
            help="Enter your Groq API key"
        )
        st.session_state.groq_api_key = groq_api_key
        
        tavily_api_key = st.text_input(
            "🌐 Tavily API Key",
            type="password",
            value=st.session_state.tavily_api_key,
            placeholder="tvly-...",
            help="Enter your Tavily API key"
        )
        st.session_state.tavily_api_key = tavily_api_key
        
        st.divider()
        
        # Model selection
        model = st.selectbox(
            "🤖 Model",
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
            index=0,
            help="Select the AI model"
        )
        
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
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching the web and thinking..."):
                response = get_ai_response(prompt, groq_api_key, tavily_api_key)
            
            # Display response
            message_placeholder = st.empty()
            message_placeholder.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
