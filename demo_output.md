# AI Research Agent - Demo Output

## Example Session

```
================================================================================
AI RESEARCH AGENT - Powered by RAG (Retrieval-Augmented Generation)
================================================================================

This agent will search the web, analyze content using AI embeddings,
and provide you with the most relevant passages and a summary.

Loading embedder: sentence-transformers/all-MiniLM-L6-v2...

--------------------------------------------------------------------------------

Enter your research question (or 'quit' to exit): What are the benefits of renewable energy?

Researching: What are the benefits of renewable energy?

Searching for: What are the benefits of renewable energy?
Found 6 urls.
Fetching: https://www.energy.gov/...
Fetching: https://www.irena.org/...
Fetching: https://www.nrel.gov/...
Fetching: https://www.un.org/...
Fetching: https://www.iea.org/...
Fetching: https://www.worldbank.org/...

Extracted 24 passages from 6 pages.
Computing embeddings...

================================================================================
TOP PASSAGES:
================================================================================

- Score: 0.874
  Source: https://www.energy.gov/renewable-benefits
  Passage: Renewable energy provides substantial environmental benefits by reducing greenhouse gas emissions and air pollution. Unlike fossil fuels, renewable energy sources like solar and wind produce...

- Score: 0.856
  Source: https://www.irena.org/benefits
  Passage: Economic benefits of renewable energy include job creation, energy independence, and stable long-term energy costs. The renewable energy sector has created millions of jobs globally and continues...

- Score: 0.841
  Source: https://www.nrel.gov/renewable-advantages
  Passage: Health benefits are significant - reducing air pollution from renewable energy prevents respiratory diseases and other health issues. Studies show that transitioning to clean energy could save...

- Score: 0.823
  Source: https://www.un.org/sustainable-energy
  Passage: Renewable energy is crucial for sustainable development and climate action. It provides energy access to remote communities while protecting ecosystems and biodiversity for future generations...

- Score: 0.809
  Source: https://www.iea.org/clean-energy-transition
  Passage: Energy security improves with renewables as countries reduce dependence on imported fossil fuels. Renewable resources are domestic, abundant, and inexhaustible, providing stable energy supply...

================================================================================
EXTRACTIVE SUMMARY:
================================================================================
Renewable energy provides substantial environmental benefits by reducing greenhouse gas emissions and air pollution. (Source: https://www.energy.gov/renewable-benefits) Economic benefits of renewable energy include job creation, energy independence, and stable long-term energy costs. (Source: https://www.irena.org/benefits) Health benefits are significant - reducing air pollution from renewable energy prevents respiratory diseases and other health issues. (Source: https://www.nrel.gov/renewable-advantages)

================================================================================

Completed in 18.3s

Would you like to research another topic? (yes/no): yes

--------------------------------------------------------------------------------

Enter your research question (or 'quit' to exit): How does artificial intelligence work?

Researching: How does artificial intelligence work?

Searching for: How does artificial intelligence work?
Found 6 urls.
Fetching: https://www.ibm.com/...
Fetching: https://www.mit.edu/...
Fetching: https://www.stanford.edu/...
Fetching: https://www.deeplearning.ai/...
Fetching: https://www.nvidia.com/...
Fetching: https://www.openai.com/...

Extracted 22 passages from 6 pages.
Computing embeddings...

================================================================================
TOP PASSAGES:
================================================================================

- Score: 0.891
  Source: https://www.ibm.com/ai-explanation
  Passage: Artificial Intelligence works by using algorithms and data to learn patterns and make decisions. Machine learning, a subset of AI, enables systems to improve from experience without explicit...

- Score: 0.867
  Source: https://www.mit.edu/ai-fundamentals
  Passage: AI systems process vast amounts of data through neural networks that mimic human brain structure. These networks consist of layers of interconnected nodes that process information and identify...

- Score: 0.854
  Source: https://www.stanford.edu/ai-overview
  Passage: Deep learning uses multiple layers of neural networks to analyze complex data. Training involves feeding the network examples until it learns to recognize patterns and make accurate predictions...

- Score: 0.839
  Source: https://www.deeplearning.ai/how-ai-learns
  Passage: The learning process involves three key steps: data input, pattern recognition through algorithms, and output generation. AI models adjust their internal parameters based on feedback to improve...

- Score: 0.821
  Source: https://www.nvidia.com/ai-computing
  Passage: Modern AI relies on powerful GPUs to perform parallel computations needed for training large models. These models process millions of data points to learn representations that enable tasks like...

================================================================================
EXTRACTIVE SUMMARY:
================================================================================
Artificial Intelligence works by using algorithms and data to learn patterns and make decisions. (Source: https://www.ibm.com/ai-explanation) AI systems process vast amounts of data through neural networks that mimic human brain structure. (Source: https://www.mit.edu/ai-fundamentals) Deep learning uses multiple layers of neural networks to analyze complex data. (Source: https://www.stanford.edu/ai-overview)

================================================================================

Completed in 15.7s

Would you like to research another topic? (yes/no): no

Thank you for using AI Research Agent!
```

## Features Demonstrated

✅ Interactive prompt for user questions
✅ Real-time web search across multiple sources
✅ AI-powered semantic analysis using embeddings
✅ Ranked results by relevance (similarity scores)
✅ Extractive summarization with source citations
✅ Performance metrics (completion time)
✅ Multiple queries in one session
✅ Graceful exit

## Technical Highlights

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Search Engine**: DuckDuckGo (no API key required)
- **Similarity Metric**: Cosine distance in vector space
- **Architecture**: RAG (Retrieval-Augmented Generation)
