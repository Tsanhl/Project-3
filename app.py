"""
🤖 AI Assistant with Specialized Law Problem Answering
A conversational AI that answers any question using web search and fact-checking
Specialized IRAC format for law questions with OSCOLA citations
Uses Groq and Tavily APIs directly for reliable performance
"""

import streamlit as st
import os
from typing import Dict, Any, List, Tuple
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🤖 AI Assistant",
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
    .law-answer {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
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

def is_law_question(query: str) -> bool:
    """Detect if a question is law-related"""
    law_keywords = [
        'law', 'legal', 'statute', 'act', 'legislation', 'court', 'judge', 'case law',
        'precedent', 'jurisdiction', 'tort', 'contract', 'negligence', 'liability',
        'claimant', 'defendant', 'plaintiff', 'breach', 'duty of care', 'damages',
        'common law', 'equity', 'judgment', 'ruling', 'appeal', 'conviction',
        'crime', 'criminal', 'civil', 'contractual', 'tortious', 'sue', 'sued',
        'litigation', 'lawsuit', 'legal advice', 'legal problem', 'legal question',
        'jurisprudence', 'legal principle', 'rule of law', 'obiter dicta', 'ratio decidendi'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in law_keywords)

def search_web(query: str, tavily_client: TavilyClient, max_results: int = 5, search_type: str = "general") -> Tuple[str, List[Dict]]:
    """Search the web using Tavily and return formatted results with source links"""
    try:
        # For law questions, prioritize legal sources
        if search_type == "law":
            # Search Google Scholar and legal databases
            enhanced_query = f"{query} site:scholar.google.com OR site:westlaw.com OR site:lexisnexis.com OR site:law.cornell.edu OR site:bailii.org OR site:legislation.gov.uk"
        else:
            enhanced_query = query
        
        response = tavily_client.search(
            query=enhanced_query,
            max_results=max_results,
            search_depth="advanced"
        )
        
        # Format results for the LLM
        formatted_results = []
        source_list = []
        
        if 'results' in response:
            for result in response['results']:
                source_info = {
                    'title': result.get('title', 'No title'),
                    'url': result.get('url', 'No URL'),
                    'content': result.get('content', 'No content')
                }
                formatted_results.append(source_info)
                source_list.append(source_info)
        
        # Create a readable summary with source links
        summary = "Web Search Results:\n\n"
        for i, result in enumerate(formatted_results, 1):
            summary += f"{i}. {result['title']}\n"
            summary += f"   URL: {result['url']}\n"
            summary += f"   Content: {result['content'][:500]}...\n\n"
        
        return (summary if formatted_results else f"No results found for: {query}", source_list)
    except Exception as e:
        return (f"Error searching web: {str(e)}", [])

def format_oscola_citation(case_name: str, year: str = "", volume: str = "", reporter: str = "", page: str = "") -> str:
    """Format a case citation in OSCOLA style"""
    # Basic OSCOLA format: Case Name [Year] Volume Reporter Page
    # Example: Entores v Miles Far East Corp [1955] 2 QB 327
    if volume and reporter and page:
        return f"{case_name} [{year}] {volume} {reporter} {page}"
    elif year:
        return f"{case_name} [{year}]"
    else:
        return case_name

def get_law_answer(query: str, groq_client: Groq, tavily_client: TavilyClient, model: str, sources: List[Dict]) -> str:
    """Generate a law answer following IRAC structure with OSCOLA citations"""
    
    # Search for additional legal sources
    law_searches = [
        f"{query} case law precedent",
        f"{query} statute legislation",
        f"{query} academic legal journal"
    ]
    
    all_sources = sources.copy()
    for search_query in law_searches[:2]:  # Limit to avoid too many API calls
        _, additional_sources = search_web(search_query, tavily_client, max_results=2, search_type="law")
        all_sources.extend(additional_sources)
    
    # Format sources for prompt
    sources_text = "\n\n".join([
        f"Source {i+1}: {s['title']}\nURL: {s['url']}\nContent: {s['content'][:300]}..."
        for i, s in enumerate(all_sources[:10])
    ])
    
    # Create comprehensive law-specific prompt
    system_prompt = """You are a legal expert AI assistant specializing in formal legal analysis. 
You MUST follow the IRAC (Issue, Rule, Analysis, Conclusion) methodology for all legal problem questions.
You MUST use OSCOLA citation format for all cases, statutes, and legal sources.
You MUST be factual, accurate, and cross-reference all legal principles with the provided sources.
All citations must be in OSCOLA format: Case Name [Year] Volume Reporter Page, or Statute Name section number.
Use formal, objective legal language. Avoid colloquialisms."""

    user_prompt = f"""You are answering a legal problem question. You MUST follow IRAC structure and OSCOLA citation format.

Legal Question: {query}

Research Sources Found:
{sources_text}

INSTRUCTIONS:
1. **ISSUE**: Identify the core legal issue(s) and sub-issues. State parties clearly.

2. **RULE**: 
   - State the relevant legal principles, rules, and tests
   - For statutes: Cite exact sections (e.g., "s.1(1) of the Occupiers' Liability Act 1957")
   - For cases: Use OSCOLA format with full citation (e.g., "Entores v Miles Far East Corp [1955] 2 QB 327")
   - All cases MUST be real and fact-checked from the sources provided
   - Include academic sources where relevant (cite properly)

3. **ANALYSIS**:
   - Link specific facts from the question to specific legal rules
   - Apply rules to facts word-by-word
   - Argue both sides (claimant/defendant perspectives)
   - State what a court would "likely hold" with justification
   - Every assertion MUST have a citation in OSCOLA format

4. **CONCLUSION**:
   - Direct answer to the issue(s)
   - Summarize without introducing new arguments
   - Advise on likely legal position and strength of case

FORMAT REQUIREMENTS:
- Use OSCOLA citations in parentheses after relevant points: (Case Name [Year] Volume Reporter Page)
- For statutes: (Act Name, s.X)
- Use headings: ## Issue, ## Rule, ## Analysis, ## Conclusion
- Be formal and objective
- Cite sources from the research provided
- If sources don't contain enough information, say so honestly

Generate your answer now following IRAC structure with proper OSCOLA citations:"""

    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more precise legal analysis
            max_tokens=4000,  # Longer responses for comprehensive legal analysis
            top_p=0.9,
        )
        
        response = completion.choices[0].message.content
        
        # Add source links section at the end
        if all_sources:
            response += "\n\n---\n\n**📚 Sources Referenced:**\n\n"
            for i, source in enumerate(all_sources[:10], 1):
                response += f"{i}. [{source['title']}]({source['url']})\n"
        
        return response
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "invalid_api_key" in error_msg.lower():
            return "❌ Error: API key authentication failed. Please check your API keys."
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            return "⚠️ Error: Rate limit exceeded. Please wait a moment and try again."
        else:
            return f"❌ Error generating legal answer: {error_msg}"

def get_general_answer(query: str, groq_client: Groq, tavily_client: TavilyClient, model: str, sources: List[Dict]) -> str:
    """Generate a general answer with source links"""
    
    sources_text = "\n\n".join([
        f"Source {i+1}: {s['title']}\nURL: {s['url']}\nContent: {s['content'][:500]}..."
        for i, s in enumerate(sources)
    ])
    
    system_prompt = """You are a helpful AI assistant that answers questions using web search results.
You provide accurate, up-to-date information based on the search results provided.
Be conversational, clear, and cite sources when relevant.
Always include source links for verifiable information.
If the search results don't contain enough information, say so honestly."""

    user_prompt = f"""Question: {query}

Web Search Results:
{sources_text}

Please provide a clear, accurate answer to the question based on the web search results above.
- Use the search results to inform your answer
- Be factual and up-to-date
- Cite sources naturally in your response
- Include relevant source links where appropriate
- If the search results don't contain enough information, say so honestly

Make your answer conversational, well-structured, and include source references:"""

    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
        )
        
        response = completion.choices[0].message.content
        
        # Add source links section at the end
        if sources:
            response += "\n\n---\n\n**📚 Sources:**\n\n"
            for i, source in enumerate(sources, 1):
                response += f"{i}. [{source['title']}]({source['url']})\n"
        
        return response
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "invalid_api_key" in error_msg.lower():
            return "❌ Error: API key authentication failed. Please check your API keys."
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            return "⚠️ Error: Rate limit exceeded. Please wait a moment and try again."
        else:
            return f"❌ Error generating answer: {error_msg}"

def get_ai_response(query: str, groq_api_key: str, tavily_api_key: str, model: str = "llama-3.1-8b-instant"):
    """Get AI response using Groq and Tavily directly with specialized handling for law questions"""
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
        
        # Detect if this is a law question
        is_law = is_law_question(query)
        search_type = "law" if is_law else "general"
        
        # Step 1: Search the web for current information
        search_status = "🔍 Searching legal databases and scholarly sources..." if is_law else "🔍 Searching the web..."
        with st.spinner(search_status):
            web_results, sources = search_web(query, tavily_client, max_results=8 if is_law else 5, search_type=search_type)
        
        # Step 2: Generate appropriate response
        if is_law:
            with st.spinner("⚖️ Analyzing legal principles and generating IRAC-structured answer with OSCOLA citations..."):
                response = get_law_answer(query, groq_client, tavily_client, model, sources)
        else:
            with st.spinner("🤖 Generating answer..."):
                response = get_general_answer(query, groq_client, tavily_client, model, sources)
        
        if not response or len(response.strip()) < 10:
            return "⚠️ Response generated but was too short. Please try again or rephrase your question."
        
        return response
        
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
        <p style="font-size: 0.9rem; color: #888;">⚖️ Law questions are automatically answered using IRAC structure with OSCOLA citations</p>
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
        
        # Model selection - using currently available Groq models
        model = st.selectbox(
            "🤖 Model",
            [
                "llama-3.1-8b-instant",  # Fast, efficient (DEFAULT)
                "llama-3.1-70b-versatile",  # More capable
                "llama-3.3-70b-versatile",  # Latest version
                "mixtral-8x7b-32768",  # Alternative option
                "gemma2-9b-it"  # Google model
            ],
            index=0,
            help="Select the AI model (llama-3.1-8b-instant is fastest and recommended)"
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
        st.info("💡 **Tip:** Law questions are automatically detected and answered using IRAC methodology with OSCOLA citations!")
    
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
        
        # Show law question indicator if detected
        is_law = is_law_question(prompt)
        if is_law:
            with st.chat_message("assistant"):
                st.info("⚖️ **Law question detected!** I'll answer using IRAC structure with OSCOLA citations.")
        
        # Generate response
        with st.chat_message("assistant"):
            response = get_ai_response(prompt, groq_api_key, tavily_api_key, model)
            
            # Display response
            message_placeholder = st.empty()
            if is_law:
                # Wrap law answers in a styled div
                message_placeholder.markdown(f'<div class="law-answer">{response}</div>', unsafe_allow_html=True)
            else:
                message_placeholder.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
