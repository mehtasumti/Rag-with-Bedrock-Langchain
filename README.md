# 🧠 RAG with AWS Bedrock & LangChain

A complete **Retrieval-Augmented Generation (RAG)** system built on AWS Bedrock, LangChain, and LangSmith — developed as part of the K21Academy *Build RAG using AWS Bedrock & SageMaker Notebook* lab.

---

## 📌 Project Overview

As an ML Engineer at MindSpace, the goal was to enhance business decision-making by building a RAG system that extracts relevant information from unstructured data sources — web pages, PDFs, and SQL databases — and generates accurate, context-aware responses using large language models.

```
User Question
     ↓
Retriever (semantic search over embedded documents)
     ↓
Relevant Context Chunks
     ↓
Prompt (context + question)
     ↓
LLM (Amazon Nova Lite via AWS Bedrock)
     ↓
Answer
```

---

## 🗂️ Repository Structure

```
Rag-with-Bedrock-Langchain/
│
├── RAG_1_Project.ipynb        # Web document RAG (IBM Think pages)
├── RAG_2_PDF_Project.ipynb    # PDF document RAG (k21.pdf)
├── RAG_3_SQL_Project.ipynb    # SQL database RAG (Chinook.db)
├── Chinook.db                 # SQLite music store database (11 tables)
├── Chinook_Sqlite.sql.txt     # Chinook database schema
└── k21.pdf                    # Sample PDF used in RAG 2
```

---

## 🧩 The Three RAG Notebooks

### 📘 RAG 1 — Web Document RAG
**File:** `RAG_1_Project.ipynb`

Loads live web pages, splits them into chunks, embeds with Amazon Titan, stores in a pure-NumPy vector store, and answers questions via semantic retrieval.

- **Data source:** IBM Think documentation (cloud computing + data science)
- **Loader:** `WebBaseLoader`
- **Chunking:** `RecursiveCharacterTextSplitter` (500 chars, 50 overlap)
- **Sample questions:** "What is cloud computing?", "What is data science?"

---

### 📗 RAG 2 — PDF Document RAG
**File:** `RAG_2_PDF_Project.ipynb`

Extracts text from a PDF file page-by-page, chunks it, embeds it, and answers questions strictly from the document context.

- **Data source:** `k21.pdf` (Amazon Q documentation)
- **Loader:** `PyPDFLoader`
- **Chunking:** `RecursiveCharacterTextSplitter` (500 chars, 50 overlap)
- **Sample questions:** "What are the business applications of Amazon Q?", "How does Amazon Q integrate with AWS?"

---

### 📕 RAG 3 — SQL Database RAG
**File:** `RAG_3_SQL_Project.ipynb`

Translates natural language business questions into SQL queries, executes them against the Chinook SQLite database, and returns human-readable answers.

- **Data source:** `Chinook.db` (11 tables: Artist, Album, Track, Customer, Invoice, Employee...)
- **Chain:** `SQLDatabaseChain` from `langchain-experimental`
- **Sample questions:** "Which employee has the most customers?", "What is the total sales by country?"

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Cloud Platform** | AWS (SageMaker Notebooks) |
| **LLM** | `amazon.nova-lite-v1:0` via AWS Bedrock |
| **Embeddings** | `amazon.titan-embed-text-v2:0` via AWS Bedrock |
| **Vector Store** | NumpyVectorStore (pure Python — replaces ChromaDB) |
| **Orchestration** | LangChain (LCEL pipe syntax) |
| **Monitoring** | LangSmith (project: K21) |
| **Database** | SQLite (Chinook) via SQLAlchemy |
| **PDF Parsing** | PyPDF via `langchain-community` |

---

## ⚙️ Setup & Installation

### Prerequisites
- AWS Account with Bedrock access enabled (`us-east-1`)
- SageMaker Notebook instance with `AmazonBedrockFullAccess` IAM policy attached
- LangSmith account at [smith.langchain.com](https://smith.langchain.com)

### 1. Clone the repository
```bash
git clone https://github.com/mehtasumti/Rag-with-Bedrock-Langchain.git
cd Rag-with-Bedrock-Langchain
```

### 2. Install dependencies
Run **Cell 1** in any notebook — it installs all required packages:
```bash
pip install langsmith langchain langchain-aws langchain-core \
            langchain-community langchain-text-splitters \
            langchain-experimental pypdf sqlalchemy
```

### 3. Configure LangSmith
In **Cell 2**, set your personal API key (generate at [smith.langchain.com/settings](https://smith.langchain.com/settings)):
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "K21"
os.environ["LANGCHAIN_API_KEY"]    = "your_lsv2_pt_key_here"
```

> ⚠️ **Use a Personal key** (`lsv2_pt_...`), not a Service/Org key (`lsv2_sk_...`). Personal keys work without workspace configuration.

### 4. Run notebooks in sequence
```
RAG_1_Project.ipynb  →  RAG_2_PDF_Project.ipynb  →  RAG_3_SQL_Project.ipynb
```

---

## 🔑 Key Design Decisions

### NumpyVectorStore instead of ChromaDB
ChromaDB requires `chroma-hnswlib`, a C++ library that needs **GCC ≥ 9.3** to compile. Amazon Linux 2 (used by SageMaker) ships with GCC 7.3.1 — making ChromaDB impossible to install. `NumpyVectorStore` is a pure-Python drop-in replacement using cosine similarity:
```python
scores = vectors @ query_vec / (norms + 1e-10)
```

### ChatBedrockConverse instead of ChatBedrock
`ChatBedrock` uses the older `InvokeModel` API. `ChatBedrockConverse` uses the newer AWS Converse API — actively maintained and recommended for all new development.

### amazon.nova-lite-v1:0 instead of claude-3-sonnet
`anthropic.claude-3-sonnet-20240229-v1:0` reached End of Life on AWS Bedrock. `amazon.nova-lite-v1:0` is the current recommended replacement — fast, cost-effective, and fully active.

---

## 📊 LangSmith Monitoring

All three notebooks send traces to the **K21** project in LangSmith:

- View at: `https://smith.langchain.com → Tracing → K21`
- Each LLM call creates a trace showing: retrieved chunks, prompt sent, tokens used, latency, and response

---

## 🗄️ Chinook Database Schema (RAG 3)

The Chinook database is a sample music store with these tables:

| Table | Description |
|---|---|
| `Artist` | Music artists |
| `Album` | Albums per artist |
| `Track` | Individual tracks |
| `Genre` | Music genres |
| `Customer` | Store customers |
| `Employee` | Store employees |
| `Invoice` | Customer purchases |
| `InvoiceLine` | Line items per invoice |
| `Playlist` | User playlists |
| `PlaylistTrack` | Tracks in playlists |
| `MediaType` | Audio formats |

---

## 🧪 Sample Outputs

**RAG 1 — Cloud Computing:**
> *"Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, analytics, and intelligence over the internet to offer faster innovation, flexible resources, and economies of scale."*

**RAG 2 — Amazon Q PDF:**
> *"Amazon Q's business applications focus on enterprise solutions, including specialized tools for enterprise clients and case studies showcasing outcomes businesses have achieved."*

**RAG 3 — SQL Query:**
```
Question: Which employee has the most customers?
SQL: SELECT e.FirstName, e.LastName, COUNT(c.CustomerId) AS CustomerCount
     FROM Employee e JOIN Customer c ON e.EmployeeId = c.SupportRepId
     GROUP BY e.EmployeeId ORDER BY CustomerCount DESC LIMIT 1;
Answer: Jane Peacock with 21 customers
```

---

## 📚 References

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [LangChain Documentation](https://python.langchain.com/docs/introduction/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [What is RAG?](https://aws.amazon.com/what-is/retrieval-augmented-generation/)

---

## 👩‍💻 Author

**Sumti Mehta** — [LinkedIn](https://www.linkedin.com/in/sumti-mehta) | [GitHub](https://github.com/mehtasumti)

*Built as part of the K21Academy AWS GenAI/ML lab series*
