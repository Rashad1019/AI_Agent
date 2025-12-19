<div align="center">
  <img src="images/python-badge.png" alt="Python 3.8+" width="400">

# 🤖 AI Research Agent - Automated Web Research with RAG

> **An intelligent research assistant that searches the web, analyzes content using AI embeddings, and delivers relevant passages with concise summaries.**

This project demonstrates a practical implementation of **Retrieval-Augmented Generation (RAG)** — the architecture powering modern AI systems like ChatGPT's web browsing capabilities. Built as a portfolio piece to showcase AI/ML engineering skills, this agent automates the time-consuming process of web research by intelligently retrieving and ranking information based on semantic similarity.

</div>

---

## 🎯 What Does It Do?

The AI Research Agent takes your research question and:
1. **Searches** the web using DuckDuckGo (no API keys required)
2. **Scrapes** and cleans content from multiple web pages
3. **Analyzes** passages using AI-powered semantic embeddings
4. **Ranks** content by relevance using cosine similarity
5. **Summarizes** findings with extractive summarization

**Example Query:** *"What causes urban heat islands and how can cities reduce them?"*

**Output:** Top 5 most relevant passages from different sources, plus a 3-sentence summary with source citations.

---

## 📹 See It In Action

**Want to see real examples before running it?**

Check out **[demo_output.md](demo_output.md)** for complete example sessions showing:

🔍 **Two Research Queries:**
1. "What are the benefits of renewable energy?"
2. "How does artificial intelligence work?"

📊 **What's Included:**
- Real-time search across 6 authoritative sources
- Top 5 passages ranked by AI similarity (scores 0.8+)
- Extractive summaries with source citations
- Performance metrics (15-18 seconds per query)
- Multiple queries in one interactive session

**Perfect for seeing** what the agent produces without setting it up first!

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/Rashad1019/AI_Agent.git
cd AI_Agent

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run the interactive research agent
python research_agent.py
```

The agent will:
- Load the AI embedding model (first run downloads ~90MB)
- Prompt you for a research question
- Search, analyze, and present results
- Ask if you want to research another topic

**Example Session:**
```
Enter your research question: How does quantum computing work?

Researching: How does quantum computing work?

TOP PASSAGES:
- Score: 0.847
  Source: https://example.com/quantum-basics
  Passage: Quantum computers use quantum bits or "qubits" that can exist in...

EXTRACTIVE SUMMARY:
Quantum computers leverage quantum mechanics principles like superposition...
(Source: https://example.com/quantum-basics)
```

---

## 📊 Business Impact & Use Cases

### Portfolio Development
- Demonstrates proficiency in AI/ML engineering
- Showcases understanding of RAG architecture
- Highlights ability to integrate multiple technologies
- Proves practical problem-solving skills

### Practical Applications
- **Students & Researchers:** Quick literature review and source discovery
- **Content Creators:** Rapid background research for articles
- **Analysts:** Market research and competitor analysis
- **Developers:** Technical documentation synthesis

### Technical Skills Demonstrated
- Natural Language Processing (NLP)
- Vector embeddings and semantic search
- Web scraping and data cleaning
- API-less search integration
- Python best practices

---

## 🏗️ Architecture Overview

```
User Query → Web Search → Content Scraping → Text Chunking
                                                    ↓
Summary ← Sentence Ranking ← Passage Ranking ← Embeddings
```

**Key Components:**
1. **Retriever:** DuckDuckGo search + BeautifulSoup scraping
2. **Ranker:** Sentence-transformers embeddings + cosine similarity
3. **Generator:** Extractive summarization from top passages

**Core Technologies:**
- `sentence-transformers` - Semantic embeddings (all-MiniLM-L6-v2 model)
- `duckduckgo-search` - Free web search without API keys
- `beautifulsoup4` - HTML parsing and text extraction
- `numpy` - Numerical computations for similarity scoring

---

## 📚 Documentation

### For Non-Technical Users
📄 **[Executive Summary](SUMMARY.md)** - Plain-language explanation of how the agent works and why it matters

### For Technical Users
💻 **[Technical Deep Dive](TECHNICAL.md)** - Architecture details, code walkthrough, and RAG implementation guide

### Live Demo
📹 **[Demo Output](demo_output.md)** - See real example queries and results

### Configuration
The agent can be customized by modifying constants in `research_agent.py`:
- `SEARCH_RESULTS = 6` - Number of URLs to analyze
- `PASSAGES_PER_PAGE = 4` - Passages extracted per page
- `TOP_PASSAGES = 5` - Results to return
- `SUMMARY_SENTENCES = 3` - Summary length
- `TIMEOUT = 8` - Webpage loading timeout (seconds)

---

## 🧪 Example Queries to Try

- "What are the latest developments in CRISPR gene editing?"
- "How do recommendation algorithms work on Netflix?"
- "What is the environmental impact of cryptocurrency mining?"
- "Explain the difference between supervised and unsupervised learning"
- "What are best practices for microservices architecture?"

---

## 🛠️ Project Structure

```
AI_Agent/
├── research_agent.py           # Main application code
├── research_agent_enhanced.py  # Enhanced version with logging
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── SUMMARY.md                  # Non-technical overview
├── TECHNICAL.md                # Technical documentation
├── QUICKSTART.md               # 5-minute setup guide
├── demo_output.md              # Example output
└── examples/                   # Sample outputs
    └── sample_research_output.txt
```

---

## 🔧 Technical Requirements

**Dependencies:**
- `ddgs` - DuckDuckGo search API wrapper
- `requests` - HTTP library
- `beautifulsoup4` - HTML parsing
- `sentence-transformers` - Embedding models
- `numpy` - Numerical operations

**System Requirements:**
- 2GB RAM minimum (4GB recommended for model loading)
- Internet connection for web search
- ~500MB disk space (including model cache)

---

## 🚦 Limitations & Future Enhancements

### Current Limitations
- English language only
- No support for PDFs or paywalled content
- Extractive (not generative) summarization
- Rate-limited by search provider

### Planned Improvements
- [ ] Add support for multiple languages
- [ ] Integrate LLM for generative summaries
- [ ] Implement caching for repeated queries
- [ ] Add PDF and academic paper support
- [ ] Create web interface with Streamlit
- [ ] Add export functionality (JSON, Markdown, PDF)
- [ ] Implement conversation history

---

## 📈 Performance Metrics

**Typical Query Performance:**
- Search: 1-2 seconds
- Scraping 6 pages: 3-5 seconds
- Embedding & ranking: 1-2 seconds
- **Total time:** 5-9 seconds per query

**Accuracy:**
- Semantic relevance: High (leverages pre-trained embeddings)
- Source diversity: Good (pulls from multiple URLs)
- Summary quality: Moderate (extractive, not generative)

---

## 👨‍💻 About This Project

This project was developed to:
1. **Expand technical portfolio** with a real-world AI application
2. **Demonstrate RAG architecture** implementation from scratch
3. **Showcase full-stack AI skills** (search, NLP, embeddings, ranking)
4. **Provide practical utility** for research automation

Built with clean, documented code following Python best practices. Ideal for demonstrating to potential employers or clients your understanding of modern AI architectures and practical engineering skills.

---

## 🤝 Contributing

This is a portfolio project, but suggestions and improvements are welcome! Feel free to:
- Open an issue for bugs or feature requests
- Fork the repo and submit pull requests
- Share how you've used or adapted the agent

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Rashad** - [GitHub](https://github.com/Rashad1019)

Project Link: [https://github.com/Rashad1019/AI_Agent](https://github.com/Rashad1019/AI_Agent)

---

## 🙏 Acknowledgments

- Special Thanks to: [Aman Kharwal](https://thecleverprogrammer.com/)
- Sentence-transformers library by UKPLab
- DuckDuckGo for free search API access
- Open-source community for the amazing tools

---

**⭐ If you find this project useful, please consider giving it a star!**
