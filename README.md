# 🚀 AI-Powered Marketplace Vendor Analysis Agent

An advanced **agentic AI system** for analyzing e-commerce vendor performance on the Olist Brazilian marketplace. This project implements multiple prompting techniques (Chain-of-Thought, Few-Shot, Evaluator-Optimizer, Voting, Self-Consistency) to provide comprehensive vendor assessments with tier classifications and actionable recommendations.

## 📋 Features

- **Multiple Agentic Techniques**: Implements 5+ prompting strategies for robust analysis
- **Streamlit Web UI**: Interactive dashboard for vendor exploration and analysis
- **CLI Support**: Batch processing for large-scale vendor analysis
- **Benchmarking System**: Evaluate and compare technique performance
- **Tier Classification**: Automatic A/B/C tier assignment based on performance metrics
- **Actionable Insights**: Generates strengths, weaknesses, and prioritized recommendations

## 🏗️ Project Structure

```
market-place-analysis-agent/
├── agent.py           # Core AI agent with multiple prompting techniques
├── app.py             # Streamlit web application
├── main_cli.py        # Command-line interface for batch processing
├── benchmark.py       # Performance evaluation and comparison tools
├── config.py          # API key and model configuration
├── data_engine.py     # Data loading and preprocessing
├── prompts.py         # LLM prompt templates
├── data/              # Olist dataset files (CSV)
├── .env               # API keys (not committed)
└── .gitignore         # Git ignore rules
```

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Groq API key (or Google Gemini API key)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Issa-Al-Alali/ai-marketplace-consultant.git
   cd ai-marketplace-consultant
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy plotly groq python-dotenv tqdm
   ```

3. **Configure API key**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Add dataset**
   
   Download the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the CSV files in the `data/` folder.

## 🚀 Usage

### Web Interface (Streamlit)

```bash
streamlit run app.py
```

Access the dashboard at `http://localhost:8501`

**Features:**
- Single vendor analysis with technique selection
- Benchmark evaluation mode
- Batch processing for multiple vendors
- Interactive visualizations and metrics

### Command Line Interface

```bash
# Single vendor analysis
python main_cli.py --mode single --vendor-id <SELLER_ID> --technique cot

# Batch processing
python main_cli.py --mode batch --batch-size 100 --technique few_shot --output results.json

# Benchmark all techniques
python main_cli.py --mode benchmark --technique all
```

**Available techniques:**
- `cot` - Chain-of-Thought
- `few_shot` - Few-Shot CoT with examples
- `evaluator` - Evaluator-Optimizer (Draft → Critique → Refine)
- `voting` - Voting ensemble

## 🤖 Agentic Techniques

| Technique | Description | LLM Calls |
|-----------|-------------|-----------|
| **Chain-of-Thought** | Step-by-step reasoning | 1 |
| **Few-Shot CoT** | Reasoning with example demonstrations | 1 |
| **Evaluator-Optimizer** | Draft → Critique → Refine loop | 3 |
| **Voting (Ensemble)** | Multiple analyses → Consensus | 3+ |
| **Self-Consistency** | Multiple reasoning paths → Converge | 3+ |
| **Comparative Analysis** | Analysis with peer benchmarks | 1 |

## 📊 Benchmark Results

The benchmark system evaluates techniques on 5 test cases of varying difficulty:

| Metric | Description |
|--------|-------------|
| Tier Accuracy | % of correct A/B/C classifications |
| Score Error | Average deviation from ground truth |
| Score ±10% | Predictions within 10 points |
| Rec Quality | Recommendation relevance score |

## 📁 Data

This project uses the **Olist Brazilian E-Commerce Dataset**, which includes:
- Customer orders and reviews
- Seller performance metrics
- Product catalog information
- Delivery tracking data

## ⚙️ Configuration

Edit `config.py` to switch between LLM providers:

```python
# Currently using Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"

# Uncomment for Google Gemini
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# MODEL_NAME = "gemini-2.5-flash"
```

## 🙏 Acknowledgments

- [Olist](https://olist.com/) for the Brazilian E-Commerce dataset
- [Groq](https://groq.com/) for LLM API access
- [Streamlit](https://streamlit.io/) for the web framework
