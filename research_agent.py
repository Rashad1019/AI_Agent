import re
import urllib.parse
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# Configuration constants
SEARCH_RESULTS = 6  # How many URLs to check
PASSAGES_PER_PAGE = 4  # How many passages to pull from each page
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_PASSAGES = 5  # How many relevant passages to return
SUMMARY_SENTENCES = 3  # How many sentences in the final summary
TIMEOUT = 8  # How long to wait for a webpage (seconds)


def unwrap_ddg(url):
    """If DuckDuckGo returns a redirect wrapper, extract the real URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        if "duckduckgo.com" in parsed.netloc:
            qs = urllib.parse.parse_qs(parsed.query)
            uddg = qs.get("uddg")
            if uddg:
                return urllib.parse.unquote(uddg[0])
    except Exception:
        pass
    return url


def search_web(query, max_results=SEARCH_RESULTS):
    """Search the web and return a list of URLs."""
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href") or r.get("url")
            if not url:
                continue
            url = unwrap_ddg(url)  # Clean up DDG redirect wrapper
            urls.append(url)
    return urls


def fetch_text(url, timeout=TIMEOUT):
    """Fetch and clean text content from a URL."""
    headers = {"User-Agent": "Mozilla/5.0 (research-agent)"}
    try:
        r = requests.get(url, timeout=timeout, headers=headers)
        if r.status_code != 200:
            return ""

        ct = r.headers.get("content-type", "")
        if "html" not in ct.lower():  # Skip non-HTML content
            return ""

        soup = BeautifulSoup(r.text, "html.parser")

        # Remove all annoying tags
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.extract()

        # Get all paragraph text
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if p])

        if text.strip():
            # Clean up whitespace
            return re.sub(r"\s+", " ", text).strip()

        # --- Fallback logic if <p> tags fail ---
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        if soup.title and soup.title.string:
            return soup.title.string.strip()

    except Exception:
        return ""  # Fail silently
    return ""


def chunk_passages(text, max_words=120):
    """Split long text into smaller passages."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words
    return chunks


def split_sentences(text):
    """A simple sentence splitter."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


class ShortResearchAgent:
    def __init__(self, embed_model=EMBEDDING_MODEL):
        print(f"Loading embedder: {embed_model}...")
        # This downloads the model on first run
        self.embedder = SentenceTransformer(embed_model)

    def run(self, query):
        start = time.time()

        # 1. Search
        print(f"\nSearching for: {query}")
        urls = search_web(query)
        print(f"Found {len(urls)} urls.")

        # 2. Fetch & Chunk
        docs = []
        for u in urls:
            print(f"Fetching: {u}")
            txt = fetch_text(u)
            if not txt:
                continue
            chunks = chunk_passages(txt, max_words=120)
            for c in chunks[:PASSAGES_PER_PAGE]:
                docs.append({"url": u, "passage": c})

        if not docs:
            print("No documents fetched.")
            return {"query": query, "passages": [], "summary": "No results found.", "time": time.time() - start}

        print(f"\nExtracted {len(docs)} passages from {len(urls)} pages.")

        # 3. Embed (Turn text into numbers)
        print("Computing embeddings...")
        texts = [d["passage"] for d in docs]
        emb_texts = self.embedder.encode(texts, convert_to_numpy=True)
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]

        # 4. Rank (Find similarity)
        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sims = [cosine(e, q_emb) for e in emb_texts]
        top_idx = np.argsort(sims)[::-1][:TOP_PASSAGES]
        top_passages = [{"url": docs[i]["url"], "passage": docs[i]["passage"], "score": sims[i]} for i in top_idx]

        # 5. Summarize (Extractive)
        sentences = []
        for tp in top_passages:
            for s in split_sentences(tp["passage"]):
                sentences.append({"sent": s, "url": tp["url"]})

        if not sentences:
            summary = "No summary could be generated."
        else:
            sent_texts = [s["sent"] for s in sentences]
            sent_embs = self.embedder.encode(sent_texts, convert_to_numpy=True)
            sent_sims = [cosine(e, q_emb) for e in sent_embs]

            top_sent_idx = np.argsort(sent_sims)[::-1][:SUMMARY_SENTENCES]
            chosen = [sentences[idx] for idx in top_sent_idx]

            # De-duplicate and format
            seen = set()
            lines = []
            for s in chosen:
                key = s["sent"].lower()[:80]  # Check first 80 chars
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"{s['sent']} (Source: {s['url']})")
            summary = " ".join(lines)

        elapsed = time.time() - start
        return {"query": query, "passages": top_passages, "summary": summary, "time": elapsed}


if __name__ == "__main__":
    print("="*80)
    print("AI RESEARCH AGENT - Powered by RAG (Retrieval-Augmented Generation)")
    print("="*80)
    print("\nThis agent will search the web, analyze content using AI embeddings,")
    print("and provide you with the most relevant passages and a summary.\n")

    agent = ShortResearchAgent()

    while True:
        print("\n" + "-"*80)
        q = input("\nEnter your research question (or 'quit' to exit): ").strip()

        if q.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using AI Research Agent!")
            break

        if not q:
            print("Please enter a valid question.")
            continue

        print(f"\nResearching: {q}\n")
        out = agent.run(q)

        print("\n" + "="*80)
        print("TOP PASSAGES:")
        print("="*80)
        for p in out["passages"]:
            print(f"\n- Score: {p['score']:.3f}")
            print(f"  Source: {p['url']}")
            print(f"  Passage: {p['passage'][:200]}...")

        print("\n" + "="*80)
        print("EXTRACTIVE SUMMARY:")
        print("="*80)
        print(out["summary"])
        print("\n" + "="*80)
        print(f"\nCompleted in {out['time']:.1f}s")

        # Ask if user wants to continue
        continue_research = input("\nWould you like to research another topic? (yes/no): ").strip().lower()
        if continue_research not in ['yes', 'y']:
            print("\nThank you for using AI Research Agent!")
            break
