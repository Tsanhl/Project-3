
# 🧠 Building an End-to-End LLM Project with RAG and AI Agents

Welcome! In this guide, we’ll walk through how to build an **end-to-end LLM project** using Retrieval-Augmented Generation (RAG), web search with **Tavily**, and intelligent **AI agents** using **CrewAI**.

**What Are Large Language Models (LLMs)?**

Large Language Models (LLMs) are advanced AI systems trained on vast amounts of text data to understand and generate human language. They utilize deep learning techniques, particularly transformer architectures, to perform tasks like translation, summarization, and question answering. These models are foundational in applications such as chatbots, virtual assistants, and content generation tools.  

**Why Do LLMs Hallucinate?**

Hallucinations in LLMs occur when the model generates information that is factually incorrect or entirely fabricated. This happens because LLMs predict text based on patterns in their training data without understanding the content. When faced with unfamiliar queries or insufficient context, they may produce plausible-sounding but inaccurate responses.  

**What Is Retrieval-Augmented Generation (RAG)?**

Retrieval-Augmented Generation (RAG) enhances LLMs by integrating real-time information retrieval. When a query is made, RAG systems fetch relevant data from external sources, such as databases or the internet, and use this information to generate more accurate and contextually relevant responses. This approach reduces hallucinations and ensures that the AI provides up-to-date information.

**What Are AI Agents?**

AI agents are software systems designed to autonomously perceive their environment, make decisions, and take actions to achieve specific goals. They can interact with users, other systems, or their environment, and are often powered by machine learning to adapt and improve over time. Unlike traditional software that follows predefined instructions, AI agents can learn from data and experiences to make informed decisions.


---

## 📘 Project Overview

This notebook demonstrates how to combine various tools and techniques to create a powerful **domain-specific question-answering system** enhanced with real-time search and collaborative AI agents similar to [perplexity.ai](https://perplexity.ai).

---

## 🔧 Key Components of the Project

### 1. Environment Setup

The project begins by installing key libraries and loading environment variables:

- [`langchain`](https://python.langchain.com) – Framework for LLM apps
- [`tavily-python`](https://pypi.org/project/tavily-python/) – Real-time search API
- `groq` – For inferencing LLM models running on groq cloud, so you don't have to run the LLM models on your local system

**Note:** Ensure you get your API keys (groq, Tavily) and add them in the jupyter notebook file.

---

### 2. Data Loading and Preprocessing

- Documents are loaded from the `crew_data` folder.
- They’re chunked using `RecursiveCharacterTextSplitter` for efficient indexing.
- Each chunk is converted into an **embedding vector** using OpenAI Embeddings.
- The embeddings are indexed with **FAISS**, a high-performance similarity search library.

> 🧠 Why RAG?  
> Retrieval-Augmented Generation lets LLMs provide more accurate answers by pulling relevant info from a custom knowledge base.

---

### 3. Retrieval-Augmented QA

The FAISS vectorstore is used to retrieve documents relevant to a user’s query. The query and context are passed to the LLM to generate a meaningful response.

```python
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

---

### 4. 🌐 Tavily Search Integration

**Tavily** lets you add real-time web search capability to your AI agents.

#### Setup

```bash
pip install tavily-python
```

```python
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
```

#### Example

```python
tavily_client.search(query="latest advancements in LLMs")
```

> 🔍 **Why Tavily?**  
> Great for retrieving up-to-date web data — ideal for dynamic and current-event-based use cases.

---

### 5. 🤖 CrewAI Agent Creation

In this project, we used CrewAI to orchestrate multiple specialized AI agents working collaboratively. Each agent is defined with a clear role, a specific goal, and a custom toolset. Here's a breakdown of the agents implemented in the notebook:

#### 1. Researcher Agent
**Role**: Researcher

**Goal**: Conduct deep research on a given query by retrieving information from both the local vectorstore and live web sources.

**Tools**: FAISS retriever (for custom document knowledge base) and Tavily search (for real-time web data)

#### 2. Writer Agent
**Role**: Content Writer

**Goal**: Generate detailed and structured responses or reports based on the information retrieved by the Researcher Agent.

**Tools**: LLM for generation, with context provided by the Researcher.

#### 3. Critic Agent
**Role**: Reviewer

**Goal**: Evaluate the response generated by the Writer Agent, check for factual accuracy, coherence, and completeness, and suggest improvements.

**Tools**: LLM for analysis and refinement tasks.

---
These agents collaborate under CrewAI’s coordination framework, where each agent autonomously performs its task and hands off results to the next agent in the workflow. This mirrors a human-like team environment, enabling a more robust and scalable AI solution.

> 💡 **Why CrewAI?**  
> CrewAI simplifies the process of designing, managing, and scaling multi-agent AI systems. It brings structure and collaboration to autonomous AI workflows, making it perfect for tasks requiring research, synthesis, and review.

Using [CrewAI](https://github.com/joaomdmoura/crewai), the project sets up multiple agents:

- Each agent has a **role**, **goal**, and **toolset** (e.g., retriever, Tavily).
- Agents can **collaborate** to achieve a shared objective (e.g., generate a report or perform research).

> This is how you simulate **team-based AI problem solving**.




---

## 🧪 Student Tasks

To deepen your understanding, try these hands-on activities:

### Task 1: Explore the Vectorstore
- Use `.similarity_search()` to see what documents match a query.
- Visualize embeddings using PCA or t-SNE.

### Task 2: Expand the Knowledge Base
- Add new documents to `crew_data`.
- Rebuild the FAISS index and test retrieval with new queries.

### Task 3: Create a New AI Agent
- Define a new role (e.g., “Trend Analyst”).
- Give it the **Tavily tool** and a goal like “Summarize the latest news on AI”.

### Task 4: Report Generator Agent
- Design an agent to collect insights and export them as a structured report (Markdown, PDF, etc.).

### Bonus Task: Compare Retrieval Methods
- Explore local LLM inferencing frameworks such as Ollama, Huggingface transformers etc instead of groq
- Replace FAISS with other vectorstores (e.g., Chroma).
- Test different chunk sizes, overlaps, and embedding models.
- Create an API using flask/django or frontend using streamlit.
- Dockerise the flask API/streamlit app for deployment.

---

## 🧵 Summary

This project gives you practical experience in:

- **Custom document QA** with RAG
- **Real-time search** using Tavily
- **Multi-agent orchestration** with CrewAI

By expanding this notebook, you'll be well on your way to building your own AI-powered assistants.

---

## 🚀 Streamlit Web Application

A comprehensive **Streamlit-based web application** has been created that provides an intuitive interface for interacting with all the AI agents.

### Features

- 🎯 **Four Operating Modes:**
  - **Research Only**: Gather information from PDF and web sources
  - **Research & Writing**: Researches and generates structured content
  - **Research, Writing & Critique**: Adds quality assurance and feedback
  - **Full Critical Analysis**: Includes deep critical thinking and evaluation

- 🤖 **Four Specialized AI Agents:**
  - **Research Agent**: Conducts comprehensive research using RAG and web search
  - **Writer Agent**: Creates well-structured, engaging written content
  - **Critic Agent**: Evaluates quality and provides constructive feedback
  - **Critical Thinking Agent**: Applies deep analytical reasoning

- 🎨 **Modern UI**: Beautiful, responsive interface with real-time progress tracking

### Quick Start

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Set Up API Keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Or enter them directly in the Streamlit sidebar when running the app.

#### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Usage Guide

1. **Configure API Keys**: Enter your Groq and Tavily API keys in the sidebar
2. **Select Mode**: Choose the desired level of analysis (Research Only, Research & Writing, etc.)
3. **Enter Query**: Type your research question or topic in the main text area
4. **Generate Response**: Click "Generate Response" and watch the AI agents work
5. **Review Results**: View the comprehensive response and download if needed
6. **History**: Access previous queries from the history section

### File Structure

```
Project 3/
├── app.py              # Streamlit web application
├── app.ipynb           # Jupyter notebook (original)
├── doc.pdf             # PDF document for RAG
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── .dockerignore      # Docker ignore patterns
└── readme.md          # This file
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t llm-research-platform .
```

### Run Docker Container

```bash
docker run -p 8501:8501 \
  -e GROQ_API_KEY=your_groq_api_key \
  -e TAVILY_API_KEY=your_tavily_api_key \
  -v $(pwd)/doc.pdf:/app/doc.pdf \
  llm-research-platform
```

Or using Docker Compose (create `docker-compose.yml`):

```yaml
version: '3.8'

services:
  streamlit-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
    volumes:
      - ./doc.pdf:/app/doc.pdf
    restart: unless-stopped
```

Then run:

```bash
docker-compose up -d
```

### Access the Application

Once running, access the application at `http://localhost:8501`

---

## 📝 API Keys Setup

### Getting a Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy and use it in the application

### Getting a Tavily API Key

1. Visit [Tavily AI](https://tavily.com/)
2. Sign up for an account
3. Navigate to API section
4. Generate your API key
5. Copy and use it in the application

---

## 🔧 Configuration Options

### Model Selection

The application supports multiple Groq models:
- `llama3-8b-8192` (default, faster)
- `llama3-70b-8192` (more powerful)
- `mixtral-8x7b-32768` (alternative option)

Select your preferred model from the sidebar.

### PDF Document

Place your PDF file (`doc.pdf`) in the project root directory, or specify a custom path in the sidebar configuration.

---

## 🎯 Use Cases

This platform is perfect for:

- 📚 **Academic Research**: Comprehensive research with citations
- ✍️ **Content Creation**: Writing articles, reports, and documents
- 🔍 **Information Analysis**: Deep analysis of complex topics
- 🧠 **Critical Thinking**: Evaluation of arguments and evidence
- 📊 **Report Generation**: Structured reports with quality assurance

---

## 🛠️ Troubleshooting

### Website Not Accessible

If the website is not loading or showing errors:

1. **Check Streamlit Installation**:
   ```bash
   pip install -r requirements.txt
   streamlit --version
   ```

2. **Verify Port Availability**:
   ```bash
   # Check if port 8501 is in use
   lsof -i :8501
   # If in use, kill the process or use a different port:
   streamlit run app.py --server.port 8502
   ```

3. **Check API Keys**:
   - Ensure both Groq and Tavily API keys are entered in the sidebar
   - Or set them in a `.env` file in the project root

4. **Check PDF File**:
   - Verify `doc.pdf` exists in the project directory
   - Or update the PDF path in the sidebar

5. **View Detailed Errors**:
   - Check the terminal/console where Streamlit is running
   - Look for error messages in red
   - The app now provides more helpful error messages

6. **Browser Issues**:
   - Clear browser cache
   - Try a different browser
   - Check browser console for errors (F12)

### Common Issues

1. **API Key Errors**: Ensure both Groq and Tavily API keys are correctly entered
2. **PDF Not Found**: Verify that `doc.pdf` exists in the project directory
3. **Import Errors**: Run `pip install -r requirements.txt` to install all dependencies
4. **Docker Issues**: Ensure Docker is running and ports are not already in use
5. **Result Not Showing**: The app now has improved result extraction. If issues persist, check terminal output for agent logs

### Accuracy & Fact-Checking

**All responses are now automatically fact-checked:**
- ✅ Every mode includes a **Fact Checker Agent** that cross-verifies with internet sources
- ✅ All facts, statistics, and claims are verified against current web data
- ✅ The Research Agent is instructed to ALWAYS use web search for current information
- ✅ Responses are marked as "Verified Response" after fact-checking

### Performance Tips

- Use `llama3-8b-8192` for faster responses
- Use `llama3-70b-8192` for more accurate, detailed responses
- "Research Only" mode is faster than "Full Critical Analysis"
- Fact-checking adds time but ensures accuracy - this is intentional

---

## 📊 Agent Workflow

### Research Only Mode
```
Query → Research Agent (Web + PDF) → Fact Checker Agent (Internet Verification) → Verified Results
```

### Research & Writing Mode
```
Query → Research Agent (Web + PDF) → Writer Agent → Fact Checker Agent (Internet Verification) → Verified Response
```

### Research, Writing & Critique Mode
```
Query → Research Agent → Writer Agent → Fact Checker Agent (Internet Verification) → Critic Agent → Verified & Refined Response
```

### Full Critical Analysis Mode
```
Query → Research Agent → Writer Agent → Fact Checker Agent (Internet Verification) → Critic Agent → Critical Thinking Agent → Comprehensive Verified Analysis
```

**Note**: All modes now include the Fact Checker Agent which cross-verifies every claim with current internet sources for 100% accuracy.

---

## 🔄 Future Enhancements

Potential improvements you can implement:

- [ ] Add support for multiple PDF documents
- [ ] Implement conversation history with chat interface
- [ ] Add export formats (PDF, Markdown, DOCX)
- [ ] Integrate additional vector stores (Chroma, Pinecone)
- [ ] Add local LLM support (Ollama, Hugging Face)
- [ ] Implement user authentication
- [ ] Add response streaming for real-time updates
- [ ] Create mobile-responsive design

---

## 📄 License

This project is provided as-is for educational purposes.

---

## 🤝 Contributing

Feel free to fork, modify, and enhance this project!

---

## 📚 Additional Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Groq API Documentation](https://console.groq.com/docs)
- [Tavily API Documentation](https://docs.tavily.com/)

---
