# 🚀 Quick Start Guide

Get up and running with the AI Research Agent in under 5 minutes!

## Prerequisites Check

Before you begin, make sure you have:
- ✅ Python 3.8 or higher
- ✅ pip (Python package manager)
- ✅ Internet connection
- ✅ ~500MB free disk space

**Check your Python version:**
```bash
python --version
# Should show: Python 3.8.x or higher
```

---

## Installation (3 Steps)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Rashad1019/AI_Agent.git
cd AI_Agent
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**What's being installed:**
- `ddgs` - DuckDuckGo search
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `sentence-transformers` - AI embeddings (~90MB download)
- `numpy` - Math operations

**Note:** First run will download the embedding model (~90MB). This is normal!

### Step 3: Run the Agent

```bash
python research_agent.py
```

**That's it!** You should see:
```
================================================================================
AI RESEARCH AGENT - Powered by RAG (Retrieval-Augmented Generation)
================================================================================
Loading embedder: sentence-transformers/all-MiniLM-L6-v2...
Enter your research question (or 'quit' to exit):
```

---

## Your First Query

### Example 1: Simple Question

**You type:**
```
What is machine learning?
```

**Agent returns (in ~7 seconds):**
- Top 5 relevant passages from different sources
- 3-sentence summary with citations
- Source links for further reading

### Example 2: Complex Question

**You type:**
```
How do neural networks learn through backpropagation?
```

**Agent returns:**
- Technical passages explaining the concept
- Summary with key insights
- Links to authoritative sources

### Example 3: Comparative Question

**You type:**
```
Python vs Java for backend development
```

**Agent returns:**
- Passages comparing both languages
- Pros/cons from different perspectives
- Summary highlighting key differences

---

## Understanding the Output

### Output Structure

```
TOP PASSAGES:
─────────────────────────────────────────────────────
Passage #1
- Score: 0.847    ← Relevance score (0-1, higher = more relevant)
- Source: [URL]   ← Where this info came from
- Passage: [Text] ← The actual content

[4 more passages...]

EXTRACTIVE SUMMARY:
─────────────────────────────────────────────────────
[3-sentence summary combining the best information]
(Source: [URL]) [Source citations included]

Time: 7.4s       ← How long it took
```

### Interpreting Scores

- **0.8 - 1.0:** Highly relevant (excellent match)
- **0.6 - 0.8:** Relevant (good match)
- **0.4 - 0.6:** Somewhat relevant (may need refinement)
- **< 0.4:** Low relevance (consider rephrasing query)

---

## Tips for Better Results

### ✅ DO:

**Be specific**
- ❌ "Tell me about AI"
- ✅ "What are the main types of machine learning algorithms?"

**Ask focused questions**
- ❌ "Everything about Python"
- ✅ "What are the advantages of Python for data science?"

**Use natural language**
- ✅ "How does photosynthesis work?"
- ✅ "What causes inflation in economics?"

### ❌ DON'T:

**Single word queries**
- ❌ "Python"
- ✅ "Python programming language features"

**Yes/no questions (unless you want explanations)**
- ❌ "Is Python good?"
- ✅ "Why is Python popular for data science?"

**Too vague**
- ❌ "Latest news"
- ✅ "Latest developments in quantum computing"

---

## Common Issues & Solutions

### Issue 1: Model Download Takes Forever

**Problem:** First run downloads 90MB model
**Solution:** 
```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Issue 2: "No documents fetched"

**Problem:** Agent couldn't extract content
**Causes:**
- Network issues
- All results were non-HTML (PDFs, etc.)
- Websites blocking scraping

**Solution:**
- Try a different query
- Check internet connection
- Try more specific keywords

### Issue 3: Low relevance scores

**Problem:** All scores < 0.5
**Solution:**
- Rephrase query to be more specific
- Add context to your question
- Try different keywords

### Issue 4: Slow performance

**Problem:** Taking > 15 seconds
**Causes:**
- Slow internet
- Unresponsive websites

**Solution:**
Edit `research_agent.py`:
```python
SEARCH_RESULTS = 3  # Reduce from 6
TIMEOUT = 5         # Reduce from 8
```

---

## Next Steps

### Option 1: Try the Enhanced Version
```bash
python research_agent_enhanced.py
```

**Benefits:**
- Better error handling
- Detailed logging
- More informative output
- Improved reliability

### Option 2: Customize Settings

Edit the configuration in `research_agent.py`:

```python
# At the top of the file:
SEARCH_RESULTS = 6        # More = broader results, slower
PASSAGES_PER_PAGE = 4     # More = more detail per source
TOP_PASSAGES = 5          # More = more results shown
SUMMARY_SENTENCES = 3     # More = longer summary
TIMEOUT = 8               # Higher = wait longer for slow sites
```

### Option 3: Read the Documentation

- **[SUMMARY.md](SUMMARY.md)** - Non-technical overview
- **[TECHNICAL.md](TECHNICAL.md)** - Deep dive into the code
- **[README.md](README.md)** - Full project documentation

### Option 4: Contribute

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for how to contribute!

---

## Example Queries to Try

### Technology
- "How does blockchain technology work?"
- "What is the difference between REST and GraphQL?"
- "Explain microservices architecture"

### Science
- "What causes the aurora borealis?"
- "How do vaccines work?"
- "Explain the theory of relativity"

### Business
- "What is the gig economy?"
- "How do recommendation algorithms work?"
- "What is agile methodology?"

### Learning
- "Best practices for Python programming"
- "How to learn machine learning?"
- "What is technical debt?"

---

## Getting Help

### If something isn't working:

1. **Check the logs** (if using enhanced version)
2. **Read the error message** (usually helpful)
3. **Try a different query** (some topics work better)
4. **Check the FAQ** (below)

### Still stuck?

- Open an issue on GitHub
- Include error messages
- Describe what you expected vs what happened

---

## FAQ

**Q: Do I need an API key?**
A: No! Everything is free and open-source.

**Q: Does this work offline?**
A: No, it needs internet to search and fetch content.

**Q: How accurate are the results?**
A: The agent finds relevant content well, but always verify important information from the original sources.

**Q: Can I use this commercially?**
A: Yes! MIT license allows commercial use.

**Q: Why is it slow sometimes?**
A: Usually due to websites taking time to respond. You can reduce `TIMEOUT` and `SEARCH_RESULTS` for speed.

**Q: Can I use it for academic research?**
A: Yes, but treat it as a starting point. Always cite original sources and verify information.

---

## What's Next?

You're all set! Start researching and exploring. The more you use it, the better you'll get at crafting effective queries.

**Happy researching! 🎉**

---

**Need more help?** Check out:
- [Full Documentation (README.md)](README.md)
- [Technical Details (TECHNICAL.md)](TECHNICAL.md)
- [Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md)
