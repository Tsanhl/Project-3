"""
🤖 Multi-Agent LLM Research & Writing Platform
A comprehensive system combining Research, Writing, Critic, and Critical Thinking agents
"""

import streamlit as st
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai import Crew, Task, Agent
import time
from datetime import datetime
from dotenv import load_dotenv

# Import tool decorator - use LangChain's tool which is compatible with CrewAI
from langchain_core.tools import tool

# Load environment variables from .env file if it exists
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🧠 AI Research & Writing Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .response-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'crew_history' not in st.session_state:
    st.session_state.crew_history = []

def initialize_llm(api_key: str, model: str = "llama3-8b-8192"):
    """Initialize the LLM with Groq"""
    return ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=api_key,
        model_name=model,
        temperature=0.1,
        max_tokens=2000,
    )

def setup_tools(groq_api_key: str, tavily_api_key: str, pdf_path: str = "doc.pdf"):
    """Setup RAG and Web Search tools"""
    # Set Tavily API key in environment (required for TavilySearchResults)
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    
    # RAG Tool for PDF search
    rag_tool = PDFSearchTool(
        pdf=pdf_path,
        config=dict(
            llm=dict(
                provider="groq",
                config=dict(
                    model="llama3-8b-8192",
                ),
            ),
            embedder=dict(
                provider="huggingface",
                config=dict(
                    model="BAAI/bge-small-en-v1.5",
                ),
            ),
        )
    )
    
    # Web Search Tool (reads TAVILY_API_KEY from environment)
    web_search_tool = TavilySearchResults(k=5)
    
    return rag_tool, web_search_tool

@tool
def router_tool(question: str) -> str:
    """Router Function to determine search method. 
    Use this to route questions to either vectorstore (PDF) or web search.
    
    Args:
        question: The user's question to route
        
    Returns:
        'vectorstore' if question is about PDF content, 'web_search' otherwise
    """
    pdf_keywords = ['Sporo', 'patient', 'chart', 'health', 'medical']
    if any(keyword.lower() in question.lower() for keyword in pdf_keywords):
        return 'vectorstore'
    else:
        return 'web_search'

def create_research_agent(llm: ChatOpenAI, rag_tool: Any, web_search_tool: Any):
    """Create Research Agent"""
    return Agent(
        role='Senior Research Analyst',
        goal='Conduct comprehensive research by gathering information from both local knowledge base (PDF) and real-time web sources to provide accurate and up-to-date information',
        backstory=(
            "You are an expert researcher with years of experience in information gathering and analysis. "
            "You excel at finding relevant information from various sources including documents, databases, and the web. "
            "You are meticulous, thorough, and always ensure the information you gather is accurate and relevant to the query."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[rag_tool, web_search_tool, router_tool]
    )

def create_writer_agent(llm: ChatOpenAI):
    """Create Writer Agent"""
    return Agent(
        role='Senior Content Writer',
        goal='Transform research findings into well-structured, coherent, and engaging written content that clearly communicates the information',
        backstory=(
            "You are an accomplished writer with expertise in creating clear, engaging, and informative content. "
            "You have a talent for organizing complex information into digestible formats. "
            "Your writing is always clear, well-structured, and tailored to the audience's needs."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

def create_critic_agent(llm: ChatOpenAI):
    """Create Critic Agent for Quality Judgment"""
    return Agent(
        role='Quality Assurance Critic',
        goal='Evaluate written content for accuracy, coherence, completeness, and quality, identifying areas for improvement',
        backstory=(
            "You are a meticulous editor and critic with years of experience in content quality assurance. "
            "You have a keen eye for detail and can identify factual inaccuracies, logical inconsistencies, "
            "and areas where content can be improved. You provide constructive feedback and ensure high standards."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

def create_critical_thinking_agent(llm: ChatOpenAI):
    """Create Critical Thinking Agent"""
    return Agent(
        role='Critical Thinking Analyst',
        goal='Apply deep analytical thinking to evaluate arguments, identify assumptions, assess evidence quality, and provide nuanced insights',
        backstory=(
            "You are a philosopher and critical thinker with expertise in logical reasoning, argument analysis, and evidence evaluation. "
            "You excel at identifying biases, questioning assumptions, evaluating the strength of evidence, "
            "and providing balanced, well-reasoned perspectives on complex topics."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

def create_fact_checker_agent(llm: ChatOpenAI, web_search_tool: Any):
    """Create Fact Checker Agent for cross-verification"""
    return Agent(
        role='Fact Verification Specialist',
        goal='Verify all factual claims in responses by cross-checking with current internet data to ensure 100% accuracy',
        backstory=(
            "You are an expert fact-checker with a meticulous approach to verification. "
            "You never accept information at face value and always cross-reference claims with multiple reliable internet sources. "
            "You specialize in identifying inaccuracies, outdated information, and unverified claims. "
            "Your primary responsibility is to ensure every fact, statistic, and claim in the response is accurate and verifiable."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[web_search_tool]
    )

def create_crew(query: str, mode: str, groq_api_key: str, tavily_api_key: str, pdf_path: str):
    """Create and configure the Crew based on mode"""
    llm = initialize_llm(groq_api_key)
    rag_tool, web_search_tool = setup_tools(groq_api_key, tavily_api_key, pdf_path)
    
    # Create agents
    research_agent = create_research_agent(llm, rag_tool, web_search_tool)
    writer_agent = create_writer_agent(llm)
    critic_agent = create_critic_agent(llm)
    critical_thinking_agent = create_critical_thinking_agent(llm)
    fact_checker_agent = create_fact_checker_agent(llm, web_search_tool)
    
    if mode == "Research Only":
        # Research-focused crew with fact-checking
        research_task = Task(
            description=f"Conduct comprehensive research on the following query: {query}. "
                       f"Use the router tool to determine whether to search the local knowledge base (PDF) or the web. "
                       f"ALWAYS use web search to gather current, accurate information. "
                       f"Then gather all relevant information from the appropriate sources. "
                       f"Provide a detailed summary of findings with sources.",
            expected_output="A comprehensive research summary with key findings, relevant information, and source references",
            agent=research_agent,
        )
        
        fact_check_task = Task(
            description=f"Verify the accuracy of all facts and claims in the research findings about '{query}'. "
                       f"Cross-check EVERY claim, statistic, and fact with current internet sources using web search. "
                       f"Identify any inaccuracies, outdated information, or unverified claims. "
                       f"Provide a verified response with corrections if needed.",
            expected_output="A fact-checked response with verified information and corrections for any inaccuracies found",
            agent=fact_checker_agent,
            context=[research_task]
        )
        
        crew = Crew(
            agents=[research_agent, fact_checker_agent],
            tasks=[research_task, fact_check_task],
            verbose=True,
        )
        
    elif mode == "Research & Writing":
        # Research + Writing crew with fact-checking
        research_task = Task(
            description=f"Conduct comprehensive research on: {query}. "
                       f"ALWAYS use web search to gather current information. "
                       f"Gather information from both local knowledge base and web sources using appropriate tools.",
            expected_output="Detailed research findings with key points and sources",
            agent=research_agent,
        )
        
        writing_task = Task(
            description=f"Based on the research findings about '{query}', write a well-structured, comprehensive article or response. "
                       f"Organize the information logically, ensure clarity, and make it engaging for the reader. "
                       f"Include source citations.",
            expected_output="A well-written, structured article or response that clearly presents the research findings with citations",
            agent=writer_agent,
            context=[research_task]
        )
        
        fact_check_task = Task(
            description=f"Verify ALL facts, statistics, and claims in the written response about '{query}'. "
                       f"Use web search to cross-check every factual assertion. "
                       f"Provide corrections for any inaccuracies and ensure the final response is 100% accurate.",
            expected_output="A fact-verified response with all claims cross-checked against internet sources",
            agent=fact_checker_agent,
            context=[writing_task]
        )
        
        crew = Crew(
            agents=[research_agent, writer_agent, fact_checker_agent],
            tasks=[research_task, writing_task, fact_check_task],
            verbose=True,
        )
        
    elif mode == "Research, Writing & Critique":
        # Full pipeline with critique and fact-checking
        research_task = Task(
            description=f"Conduct comprehensive research on: {query}. "
                       f"ALWAYS use web search to verify current information.",
            expected_output="Detailed research findings with key points and verified sources",
            agent=research_agent,
        )
        
        writing_task = Task(
            description=f"Based on the research about '{query}', write a comprehensive, well-structured response.",
            expected_output="A well-written response presenting the research findings",
            agent=writer_agent,
            context=[research_task]
        )
        
        fact_check_task = Task(
            description=f"Verify ALL facts and claims in the written response about '{query}'. "
                       f"Cross-check every assertion with web search. Correct any inaccuracies.",
            expected_output="A fact-verified response with corrections for inaccuracies",
            agent=fact_checker_agent,
            context=[writing_task]
        )
        
        critique_task = Task(
            description=f"Review and critique the fact-checked response about '{query}'. "
                       f"Evaluate its accuracy, coherence, completeness, clarity, and overall quality. "
                       f"Ensure all facts have been verified. "
                       f"Identify any remaining issues and provide specific, constructive feedback.",
            expected_output="A detailed critique with specific feedback on quality, accuracy, and suggestions for improvement",
            agent=critic_agent,
            context=[writing_task, fact_check_task]
        )
        
        crew = Crew(
            agents=[research_agent, writer_agent, fact_checker_agent, critic_agent],
            tasks=[research_task, writing_task, fact_check_task, critique_task],
            verbose=True,
        )
        
    else:  # Full Critical Analysis
        # Complete pipeline with critical thinking and fact-checking
        research_task = Task(
            description=f"Conduct comprehensive research on: {query}. "
                       f"ALWAYS use web search to gather current, accurate information from multiple sources.",
            expected_output="Detailed research findings from multiple verified sources",
            agent=research_agent,
        )
        
        writing_task = Task(
            description=f"Based on the research about '{query}', write a comprehensive, well-structured response with citations.",
            expected_output="A well-written response with source citations",
            agent=writer_agent,
            context=[research_task]
        )
        
        fact_check_task = Task(
            description=f"CRITICALLY verify ALL facts, statistics, dates, names, and claims in the response about '{query}'. "
                       f"Use web search to cross-check EVERY factual assertion against current internet sources. "
                       f"Identify and correct any inaccuracies, outdated information, or unverified claims. "
                       f"Ensure 100% factual accuracy.",
            expected_output="A completely fact-verified response with all corrections applied and verified sources",
            agent=fact_checker_agent,
            context=[writing_task]
        )
        
        critique_task = Task(
            description=f"Review and critique the fact-verified response about '{query}'. "
                       f"Evaluate its accuracy, coherence, completeness, clarity, and overall quality. "
                       f"Ensure all facts have been properly verified. Provide specific, constructive feedback.",
            expected_output="A detailed critique confirming factual accuracy and quality",
            agent=critic_agent,
            context=[writing_task, fact_check_task]
        )
        
        critical_analysis_task = Task(
            description=f"Apply deep critical thinking to analyze the verified response about '{query}'. "
                       f"Evaluate the arguments presented, identify underlying assumptions, assess the quality and reliability of evidence, "
                       f"identify potential biases, and provide a balanced, nuanced analysis. "
                       f"Consider alternative perspectives and the strength of the reasoning. "
                       f"Ensure all claims are backed by verified internet sources.",
            expected_output="A deep critical analysis with evaluation of arguments, assumptions, evidence quality, and alternative perspectives, all verified against internet sources",
            agent=critical_thinking_agent,
            context=[writing_task, fact_check_task, critique_task]
        )
        
        crew = Crew(
            agents=[research_agent, writer_agent, fact_checker_agent, critic_agent, critical_thinking_agent],
            tasks=[research_task, writing_task, fact_check_task, critique_task, critical_analysis_task],
            verbose=True,
        )
    
    return crew

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 AI Research & Writing Platform</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem;">Powered by Multi-Agent AI: Research • Writing • Critique • Critical Thinking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys (load from .env or user input)
        default_groq = os.getenv("GROQ_API_KEY", "")
        default_tavily = os.getenv("TAVILY_API_KEY", "")
        
        groq_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            value=default_groq,
            help="Enter your Groq API key for LLM inference (or set GROQ_API_KEY in .env)"
        )
        
        tavily_api_key = st.text_input(
            "🌐 Tavily API Key",
            type="password",
            value=default_tavily,
            help="Enter your Tavily API key for web search (or set TAVILY_API_KEY in .env)"
        )
        
        st.divider()
        
        # Mode selection
        mode = st.selectbox(
            "🎯 Select Mode",
            [
                "Research Only",
                "Research & Writing",
                "Research, Writing & Critique",
                "Full Critical Analysis"
            ],
            help="Choose the level of analysis you need"
        )
        
        # Mode descriptions
        mode_descriptions = {
            "Research Only": "🔍 Gathers information and verifies facts with web sources",
            "Research & Writing": "📝 Researches, writes, and verifies all facts with internet data",
            "Research, Writing & Critique": "✍️ Full pipeline with fact-checking and quality critique",
            "Full Critical Analysis": "🧠 Complete analysis with fact-verification and critical thinking"
        }
        
        st.info(mode_descriptions[mode])
        
        st.divider()
        
        # Model selection
        model = st.selectbox(
            "🤖 Model",
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
            index=0
        )
        
        st.divider()
        
        # PDF path
        pdf_path = st.text_input(
            "📄 PDF Path",
            value="doc.pdf",
            help="Path to your PDF document for RAG"
        )
        
        st.divider()
        
        # Agent descriptions
        st.header("🤖 Agents")
        with st.expander("Research Agent"):
            st.write("Gathers information from local PDF and web sources")
        with st.expander("Writer Agent"):
            st.write("Creates well-structured, engaging content")
        with st.expander("Fact Checker Agent"):
            st.write("✅ Verifies all facts by cross-checking with internet sources")
        with st.expander("Critic Agent"):
            st.write("Evaluates quality and provides feedback")
        with st.expander("Critical Thinking Agent"):
            st.write("Applies deep analytical reasoning")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Your Query")
        query = st.text_area(
            "Enter your research question or topic:",
            height=100,
            placeholder="e.g., What are the latest advancements in AI? Or: Does Sporo streamline patient chart reviews?"
        )
        
        # Submit button
        submit_button = st.button("🚀 Generate Response", type="primary", use_container_width=True)
    
    with col2:
        st.header("📊 Features")
        st.markdown("""
        - 🔍 **Intelligent Research**: RAG + Web Search
        - ✍️ **Quality Writing**: Structured content generation
        - ✅ **Fact Verification**: Cross-checks all facts with internet data
        - 🎯 **Smart Critique**: Quality assurance & feedback
        - 🧠 **Critical Analysis**: Deep reasoning & evaluation
        """)
    
    # Process query
    if submit_button and query:
        if not groq_api_key or not tavily_api_key:
            st.error("❌ Please enter both API keys in the sidebar!")
            return
        
        # Display progress
        with st.spinner("🤖 AI agents are working on your query..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔍 Initializing agents...")
                progress_bar.progress(10)
                
                # Create crew
                crew = create_crew(query, mode, groq_api_key, tavily_api_key, pdf_path)
                
                status_text.text("🧠 Agents are processing your query...")
                progress_bar.progress(30)
                
                # Execute crew
                start_time = time.time()
                result = crew.kickoff(inputs={"query": query})
                elapsed_time = time.time() - start_time
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                # Extract result text properly (handle CrewAI result object)
                result_text = ""
                
                # Try multiple ways to extract the result
                if hasattr(result, 'raw') and result.raw:
                    result_text = str(result.raw)
                elif hasattr(result, 'content') and result.content:
                    result_text = str(result.content)
                elif hasattr(result, 'tasks_output') and result.tasks_output:
                    # Get the last task's output (which should be the final result)
                    if isinstance(result.tasks_output, list) and len(result.tasks_output) > 0:
                        result_text = str(result.tasks_output[-1])
                    else:
                        result_text = str(result.tasks_output)
                elif isinstance(result, dict):
                    result_text = result.get('output', result.get('result', str(result)))
                elif hasattr(result, '__dict__'):
                    # Try to get attributes
                    if hasattr(result, 'output'):
                        result_text = str(result.output)
                    elif hasattr(result, 'result'):
                        result_text = str(result.result)
                    else:
                        result_text = str(result)
                else:
                    result_text = str(result)
                
                # Clean up the result text
                result_text = result_text.strip()
                
                # Ensure we have valid content
                if not result_text or len(result_text) < 10:
                    # Last resort - convert entire result to string
                    result_text = str(result)
                    if len(result_text.strip()) < 10:
                        result_text = "⚠️ Response generated but content extraction failed. The crew executed successfully. Please check the console/terminal for detailed agent output."
                
                # Store in history
                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "mode": mode,
                    "response": result_text,
                    "time": f"{elapsed_time:.2f}s"
                }
                st.session_state.crew_history.append(history_entry)
                
                # Display results
                st.success(f"✅ Completed in {elapsed_time:.2f} seconds!")
                
                st.divider()
                st.header("📝 Verified Response")
                st.info("✨ All facts have been cross-checked with internet sources for accuracy")
                st.markdown(f'<div class="response-box">{result_text}</div>', unsafe_allow_html=True)
                
                # Download option
                st.download_button(
                    label="📥 Download Response",
                    data=result_text,
                    file_name=f"verified_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error: {error_msg}")
                
                # More helpful error messages
                if "API key" in error_msg or "authentication" in error_msg.lower():
                    st.warning("💡 Please check your API keys in the sidebar. Ensure they are valid and have proper permissions.")
                elif "PDF" in error_msg or "file" in error_msg.lower():
                    st.warning(f"💡 Please verify that '{pdf_path}' exists in the project directory.")
                elif "timeout" in error_msg.lower():
                    st.warning("💡 Request timed out. Try using a faster model or simplifying your query.")
                else:
                    with st.expander("🔍 Detailed Error Information"):
                        st.exception(e)
                
                # Log to history for debugging
                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "mode": mode,
                    "response": f"ERROR: {error_msg}",
                    "time": "N/A"
                }
                st.session_state.crew_history.append(history_entry)
    
    # History section
    if st.session_state.crew_history:
        st.divider()
        st.header("📜 Query History")
        
        for idx, entry in enumerate(reversed(st.session_state.crew_history[-10:])):  # Show last 10
            with st.expander(f"Query {len(st.session_state.crew_history) - idx}: {entry['query'][:50]}..."):
                st.write(f"**Mode:** {entry['mode']}")
                st.write(f"**Time:** {entry['timestamp']} ({entry['time']})")
                st.write(f"**Response:**")
                st.text(entry['response'])
        
        if st.button("🗑️ Clear History"):
            st.session_state.crew_history = []
            st.rerun()

if __name__ == "__main__":
    main()
