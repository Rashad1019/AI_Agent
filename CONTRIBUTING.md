# Contributing to AI Research Agent

Thank you for considering contributing to the AI Research Agent! This document provides guidelines for contributing to the project.

## 🎯 Ways to Contribute

### 1. Report Bugs
Found a bug? Please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)

### 2. Suggest Features
Have an idea? Open an issue with:
- Description of the feature
- Why it would be useful
- Possible implementation approach

### 3. Improve Documentation
- Fix typos or unclear explanations
- Add examples
- Improve code comments
- Create tutorials

### 4. Submit Code
- Bug fixes
- Performance improvements
- New features
- Tests

## 🚀 Getting Started

### Fork and Clone
```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/AI_Agent.git
cd AI_Agent

# Add upstream remote
git remote add upstream https://github.com/Rashad1019/AI_Agent.git
```

### Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest black flake8
```

### Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## 📝 Code Standards

### Python Style
- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused (single responsibility)

### Example:
```python
def fetch_text(url, timeout=TIMEOUT):
    """
    Fetch and clean text content from a URL.
    
    Args:
        url: The URL to fetch content from
        timeout: Request timeout in seconds
        
    Returns:
        Cleaned text content or empty string on failure
    """
    # Implementation
    pass
```

## 📤 Submitting Changes

### Commit Messages
Follow conventional commits format:
```
feat: add parallel web fetching
fix: handle timeout errors gracefully
docs: improve README installation section
refactor: extract embedding logic
test: add tests for chunking function
```

### Push and Create PR
```bash
# Commit your changes
git add .
git commit -m "feat: add your feature"

# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

## 🎨 Areas Looking for Contributions

### High Priority
- [ ] Performance optimization (parallel fetching)
- [ ] Better error handling and logging
- [ ] Comprehensive test suite
- [ ] GPU acceleration support

### Medium Priority
- [ ] Web interface (Streamlit/Gradio)
- [ ] Result caching system
- [ ] Academic paper search integration
- [ ] Export functionality (PDF, DOCX)

### Nice to Have
- [ ] Multi-language support
- [ ] Conversation memory
- [ ] Abstractive summarization with LLMs
- [ ] Docker containerization

## 🙏 Recognition

Contributors will be:
- Listed in README.md
- Credited in release notes
- Appreciated forever! ✨

## 📞 Contact

- **GitHub Issues**: For bugs and features
- **Discussions**: For questions and ideas

---

**Thank you for contributing to making research faster and more accessible!**
