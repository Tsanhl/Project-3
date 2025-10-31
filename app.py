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

def comprehensive_search(query: str, tavily_client: TavilyClient, search_type: str = "general") -> List[Dict]:
    """Perform comprehensive multi-query search for maximum accuracy"""
    all_sources = []
    seen_urls = set()  # Avoid duplicates
    
    # For law questions, do multiple specialized searches
    if search_type == "law":
        search_queries = [
            query,  # Original query
            f"{query} case law UK",  # UK case law
            f"{query} statute legislation",  # Statutes
            f"{query} legal precedent",  # Precedents
            f"{query} academic legal journal",  # Academic sources
            f"{query} court decision",  # Court decisions
            f"{query} legal principle"  # Legal principles
        ]
    else:
        # For general questions, use variations
        search_queries = [
            query,  # Original query
            f"{query} facts",  # Factual information
            f"{query} latest",  # Latest information
            f"{query} accurate"  # Verified information
        ]
    
    # Perform multiple searches
    for search_query in search_queries[:5]:  # Limit to 5 searches to avoid rate limits
        try:
            _, sources = search_web(search_query, tavily_client, max_results=5, search_type=search_type)
            for source in sources:
                if source['url'] not in seen_urls:
                    all_sources.append(source)
                    seen_urls.add(source['url'])
        except Exception as e:
            continue  # Continue with other searches if one fails
    
    return all_sources[:20]  # Return up to 20 unique sources

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

def get_law_answer(query: str, groq_client: Groq, tavily_client: TavilyClient, model: str, initial_sources: List[Dict]) -> str:
    """Generate a law answer following IRAC structure with OSCOLA citations and comprehensive fact-checking"""
    
    # Perform comprehensive multi-source search
    with st.spinner("🔍 Performing comprehensive legal database search..."):
        all_sources = comprehensive_search(query, tavily_client, search_type="law")
        # Merge with initial sources
        seen_urls = {s['url'] for s in all_sources}
        for source in initial_sources:
            if source['url'] not in seen_urls:
                all_sources.append(source)
                seen_urls.add(source['url'])
    
    # Remove duplicates and limit to most relevant sources
    all_sources = all_sources[:15]
    
    # Format sources with full content for better context
    sources_text = "\n\n".join([
        f"=== Source {i+1}: {s['title']} ===\nURL: {s['url']}\nContent: {s['content'][:600]}\n"
        for i, s in enumerate(all_sources)
    ])
    
    # Create comprehensive law-specific prompt with strict accuracy requirements
    system_prompt = """You are a legal expert AI assistant specializing in formal legal analysis with STRICT ACCURACY requirements.
CRITICAL RULES:
1. You MUST ONLY state facts, cases, statutes, or legal principles that are EXPLICITLY found in the provided sources
2. You MUST cross-reference information across multiple sources before stating it as fact
3. If multiple sources conflict, you MUST state this conflict explicitly
4. If information is not found in sources, you MUST state "The sources do not provide sufficient information on this point"
5. You MUST follow IRAC (Issue, Rule, Analysis, Conclusion) methodology
6. You MUST use OSCOLA citation format for ALL cases and statutes
7. ALL citations must be VERIFIED from the sources - NO FABRICATED CITATIONS
8. Use formal, objective legal language
9. If unsure about any fact, state the uncertainty clearly"""

    user_prompt = f"""CRITICAL: Answer this legal question using ONLY information found in the sources below. DO NOT make up facts, cases, or citations. ALL cases, statutes, and legal principles MUST be REAL and found in the sources.

Legal Question: {query}

=== RESEARCH SOURCES (Cross-reference these carefully - ALL must be REAL) ===
{sources_text}

=== STRICT INSTRUCTIONS ===

**CRITICAL ACCURACY REQUIREMENTS:**
- BEFORE stating any fact, case name, statute, or legal principle, VERIFY it appears EXACTLY in the sources above
- If a case name is not in the sources, DO NOT cite it - DO NOT INVENT CASES
- If a statute section is not mentioned in the sources, DO NOT cite it - DO NOT INVENT STATUTES
- ALL case names, statute names, and legal principles MUST be extracted EXACTLY as they appear in the sources
- Cross-check information across multiple sources for consistency
- If sources contradict each other, acknowledge the contradiction explicitly
- DO NOT use any information not found in the sources - this is MANDATORY

**IRAC STRUCTURE:**

Issue
- DO NOT use ## or bold formatting for "Issue"
- Identify core legal issue(s) and sub-issues clearly
- State parties clearly (claimant, defendant, etc.)
- ONLY identify issues that can be addressed with information from sources
- Be precise and specific

Rule
- State legal principles found in the sources with FULL OSCOLA citations
- For statutes: Cite exact sections ONLY if found in sources with full citation format: (Act Name, s.X(subsection))
  - Example: (Occupiers' Liability Act 1957, s.2(1))
  - MUST include the exact section number as it appears in sources
- For cases: Use COMPLETE OSCOLA format ONLY if the case is mentioned in sources
  - Format: (Case Name [Year] Volume Reporter Page)
  - Example: (Entores v Miles Far East Corp [1955] 2 QB 327)
  - If full citation is in sources, use it exactly as provided
  - If only partial citation, use what is provided but note if incomplete
- Include academic sources where found in sources with proper citation
- If a legal principle is not in sources, state: "The sources do not provide the legal rule on this point"
- EVERY rule statement MUST have a citation from the sources

Analysis
- This is the MOST CRITICAL section - analyze word-by-word, fact-by-fact
- NO MISSING DETAILS - examine every fact in the question carefully
- Link EACH specific fact from the question to specific legal rules found in sources
  - Quote the exact fact: "The fact that [quote exact words from question] relates to [legal rule]"
  - Apply rules from sources to facts with precision
- Apply rules to facts systematically:
  - Take each fact word-by-word
  - Identify which legal rule applies (with citation)
  - Explain how the fact satisfies or fails to satisfy each element of the rule
- Argue both sides using principles from sources:
  - Claimant's argument: [specific argument with source citation]
  - Defendant's counter-argument: [specific counter-argument with source citation]
- Analyze every element of each legal test:
  - If duty of care: analyze proximity, foreseeability, fairness (with citations)
  - If breach: analyze reasonable person standard, factual circumstances (with citations)
  - If causation: analyze factual causation, legal causation (with citations)
  - Continue for ALL elements
- State what a court would likely hold based on sources, with justification
- Every single assertion MUST have a FULL OSCOLA citation from sources
- Use specific quotations from the question: "The question states '[exact quote]', which indicates..."
- If analysis requires information not in sources, state this limitation explicitly
- Cross-reference multiple sources when analyzing complex points

Conclusion
- Direct answer to the issue(s) based on sources
- Summarize without introducing new arguments
- Advise on likely legal position based on available sources with citation support
- If conclusion is limited by missing information, state this clearly

**CITATION FORMAT (MANDATORY):**
- Use FULL OSCOLA citations in parentheses for EVERY legal point
- Cases: (Case Name [Year] Volume Reporter Page) - MUST be from sources
- Statutes: (Act Name, s.X) - MUST be from sources
- Academic sources: (Author, "Title" [Year] Journal Volume Page) - MUST be from sources
- Reference source by number when appropriate: (Source 1), (Source 3)
- NO CITATIONS to cases/statutes not found in sources - this is PROHIBITED

**FACT ANALYSIS REQUIREMENTS:**
- Read the question word-by-word
- Identify EVERY fact stated or implied
- Analyze EACH fact against relevant legal rules
- Quote exact wording from the question when making points
- Do not skip or miss any details
- If a fact is ambiguous, identify both interpretations and analyze both
- Show how each fact affects the legal analysis

**IF INFORMATION IS MISSING:**
- If sources don't contain sufficient information on a legal point, explicitly state: "The available sources do not provide sufficient information to address [specific point]"
- DO NOT speculate or use general knowledge beyond what is in sources
- Be transparent about limitations in analysis

Generate your answer now, ensuring:
1. EVERY claim is verified against the sources
2. ALL citations are REAL and from sources
3. Analysis is word-by-word, fact-by-fact with no missing details
4. Issue section has no ## or bold formatting
5. Rule and Analysis have FULL OSCOLA citations for every legal point"""

    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Very low temperature for maximum accuracy and precision
            max_tokens=4000,  # Longer responses for comprehensive legal analysis
            top_p=0.8,  # Lower top_p for more focused, accurate responses
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

def get_general_answer(query: str, groq_client: Groq, tavily_client: TavilyClient, model: str, initial_sources: List[Dict]) -> str:
    """Generate a general answer with comprehensive fact-checking and source links"""
    
    # Perform comprehensive multi-source search
    with st.spinner("🔍 Gathering comprehensive information..."):
        all_sources = comprehensive_search(query, tavily_client, search_type="general")
        # Merge with initial sources
        seen_urls = {s['url'] for s in all_sources}
        for source in initial_sources:
            if source['url'] not in seen_urls:
                all_sources.append(source)
                seen_urls.add(source['url'])
    
    # Remove duplicates and limit to most relevant sources
    all_sources = all_sources[:12]
    
    sources_text = "\n\n".join([
        f"=== Source {i+1}: {s['title']} ===\nURL: {s['url']}\nContent: {s['content'][:600]}\n"
        for i, s in enumerate(all_sources)
    ])
    
    system_prompt = """You are a helpful AI assistant that provides ACCURATE, FACT-CHECKED answers using web search results.
CRITICAL RULES:
1. You MUST ONLY state facts that are EXPLICITLY found in the provided sources
2. You MUST cross-reference information across multiple sources for accuracy
3. If sources conflict, acknowledge the conflict
4. If information is not in sources, explicitly state this
5. Be conversational but accurate
6. Always cite which source supports each claim
7. If unsure, state your uncertainty clearly"""

    user_prompt = f"""CRITICAL: Answer this question using ONLY information found in the sources below. Cross-reference sources for accuracy.

Question: {query}

=== RESEARCH SOURCES (Cross-reference these for accuracy) ===
{sources_text}

=== INSTRUCTIONS ===
1. **ACCURACY FIRST**: 
   - ONLY state facts that appear in the sources above
   - Cross-check information across multiple sources
   - If sources disagree, state the different perspectives
   - If information is missing, say "The available sources do not provide information on this point"

2. **STRUCTURE**:
   - Provide a clear, well-structured answer
   - Cite sources naturally (e.g., "According to Source 1..." or "[Source Title](URL)")
   - Include specific facts, numbers, dates from sources when available
   - Be conversational but precise

3. **TRANSPARENCY**:
   - If you cannot answer fully based on sources, state this clearly
   - If sources conflict, acknowledge the conflict
   - Don't speculate beyond what sources provide

4. **SOURCE REFERENCES**:
   - Reference sources by number or title
   - Include clickable links where relevant
   - At the end, list all sources used

Generate your answer now, ensuring EVERY claim is verified against the sources:"""

    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,  # Lower temperature for more accurate general answers
            max_tokens=2500,  # Longer responses for comprehensive answers
            top_p=0.9,  # Slightly lower for more focused responses
        )
        
        response = completion.choices[0].message.content
        
        # Add source links section at the end
        if all_sources:
            response += "\n\n---\n\n**📚 Sources:**\n\n"
            for i, source in enumerate(all_sources, 1):
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
        
        # Step 1: Initial search for current information
        search_status = "🔍 Searching legal databases and scholarly sources..." if is_law else "🔍 Searching the web..."
        with st.spinner(search_status):
            web_results, initial_sources = search_web(query, tavily_client, max_results=8 if is_law else 5, search_type=search_type)
        
        # Step 2: Generate appropriate response with comprehensive fact-checking
        if is_law:
            with st.spinner("⚖️ Cross-referencing legal sources and generating verified IRAC-structured answer..."):
                response = get_law_answer(query, groq_client, tavily_client, model, initial_sources)
        else:
            with st.spinner("🤖 Cross-checking facts and generating verified answer..."):
                response = get_general_answer(query, groq_client, tavily_client, model, initial_sources)
        
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
