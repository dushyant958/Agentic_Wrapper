# 🧠 Agentic Wrapper (MVP)

**Agentic Wrapper** is a modular, backend-first **AI assistant framework** — built entirely from scratch — that combines **chat, document intelligence, research automation, and multi-agent orchestration** into one unified system.

It is designed as a **lightweight, customizable alternative to tools like Perplexity**, showing how a full agentic reasoning system can be engineered from the ground up using **FastAPI**, **Streamlit**, **CrewAI**, and **Groq-hosted LLaMA models**.

> ⚠️ This is the **MVP (Minimum Viable Product)** version.  
> It focuses on backend logic, architecture, and extensibility rather than frontend polish.

---

## 🚀 Core Idea

The main goal of Agentic Wrapper is to demonstrate that a **robust AI agentic framework** — capable of chatting, reasoning, researching, summarizing, and interpreting uploaded data — can be **built fully from scratch** using open frameworks and modular components.

This project integrates:
- **LLMs (via Groq)** for reasoning and text generation  
- **CrewAI** for multi-agent orchestration  
- **RAG (Retrieval-Augmented Generation)** for document & web search  
- **Whisper** for voice transcription (planned)  
- **Streamlit** for a lightweight, clean user interface  

---

## ✨ Features (MVP)

| Feature | Description |
|----------|-------------|
| 💬 **Chat Mode** | Natural, contextual chatbot using Groq-hosted LLaMA models with list-based memory for session context. |
| 🧠 **Agentic Research Mode** | Multi-agent system using CrewAI — Planner, Researcher, Verifier, and Writer agents collaborate to produce deep research reports. |
| 📄 **Document RAG** | Upload PDFs or images to query and extract relevant insights using EmbedChain and EasyOCR. |
| 🗣️ **Voice Input (Upcoming)** | Speak your query; Whisper transcribes audio to text, which is processed like a chat query. |
| 🌐 **Web Search Integration** | Serper or Google Custom Search API used for retrieving fresh, factual information. |
| ⚙️ **Dynamic Model Selection** | User can choose between Groq-hosted LLaMA variants — `8B-Instant`, `70B-Versatile`, or `Llama-Guard`. |
| 🧾 **Summarizer (Long Context Handler)** | Automatically condenses older chat messages when token limits are reached. |
| 🔄 **FastAPI Backend** | All processing handled by asynchronous API endpoints for performance and concurrency. |
| 🖥️ **Streamlit Frontend** | Clean, minimal frontend interface for chatting, uploading files, or running agentic research tasks. |

---

## 🧩 System Architecture

### 🧠 1. Core Intelligence Layer (CrewAI System)
- `agents.py` — defines the AI agents and their roles  
- `tasks.py` — defines the agent workflows and task sequence  
- `tools.py` — defines the tools (web search, RAG, scraper, etc.)  
- `crew.py` — orchestrates agents and tasks into one unified pipeline  

Agents (Planner, Researcher, Verifier, Writer) use the selected LLM model and can delegate tasks dynamically through CrewAI’s orchestration engine.

---

### ⚙️ 2. FastAPI Backend (Service Layer)
Handles:
- Chat requests (`/chat`)
- Multi-agent research pipeline (`/research`)
- File uploads (`/upload`)
- Voice transcription (`/transcribe`)
- Long-context summarization (`/summarize`)

Each endpoint runs asynchronously for speed and returns structured JSON responses for Streamlit.

---

### 🖥️ 3. Streamlit Frontend (User Interface Layer)
Provides a simple, interactive web UI with:
- Chat history and input field  
- Model selection dropdown  
- Buttons for switching between *Chat* and *Research* modes  
- File upload for PDFs and images  
- (Future) Microphone input for voice queries  

Session context (chat history) is maintained using `st.session_state` and stored as a **list** during MVP.

---

### 📚 4. Knowledge & Context Layer
- **List-based memory:** maintains short-term chat history  
- **Summarizer model:** condenses long histories when approaching token limits  
- **Persistent database memory:** will be added later for user-specific recall  

---

### 🧾 5. File Intelligence Layer
- PDFs processed with `pdfplumber`  
- Images processed with `easyocr`  
- Extracted text embedded via `EmbedChain` and stored in FAISS vector store for semantic retrieval during queries  

---

### 🔊 6. Voice Intelligence (Planned)
- Integration with **Whisper** or **Faster-Whisper** for real-time speech-to-text conversion  
- Transcribed text automatically submitted as a chat message  

---

### ⚙️ 7. Backend Orchestration Flow

```bash
User → Streamlit → FastAPI → CrewAI Agents → Tools → Groq Models → Response → Streamlit
````

Each request is handled through a REST API call:

* `/chat` → Simple model-based Q&A
* `/research` → Multi-agent CrewAI workflow
* `/upload` → RAG extraction pipeline
* `/transcribe` → Whisper transcription
* `/summarize` → Context management

---

## 🧱 Project Structure

```
Agentic-Wrapper/
│
├── backend/
│   ├── app.py                 # FastAPI backend entry point
│   ├── agents.py              # Defines Planner, Researcher, Verifier, Writer agents
│   ├── tasks.py               # Defines multi-agent workflows
│   ├── tools.py               # Defines scraping, RAG, and API tools
│   ├── crew.py                # Orchestrates agents and tasks
│   ├── utils/
│   │   ├── summarizer.py      # Summarization and context management
│   │   ├── rag_pipeline.py    # PDF/Image RAG logic
│   │   └── transcription.py   # Whisper integration (planned)
│   └── requirements.txt
│
├── frontend/
│   ├── streamlit_app.py       # Streamlit-based UI
│   ├── assets/                # Logos, icons, styles
│   └── components/            # Optional modular UI components
│
└── README.md
```

---

## 🧠 LLM Model Selection

The user can select which LLM to use through the interface.
Supported models (via **Groq API**):

| Model                          | Description                                |
| ------------------------------ | ------------------------------------------ |
| `llama-3.1-8b-instant`         | Fastest for conversational queries         |
| `llama-3.3-70b-versatile`      | High-quality reasoning and analysis        |
| `meta-llama/llama-guard-4-12b` | For safety, filtering, or moderation tasks |

The LLM initialization is dynamically handled by:

```python
def initialise_llms(self, llm_choice: str = None):
    llm_choice = llm_choice or "llama-3.1-8b-instant"
    llm_options = {
        "llama-3.1-8b-instant": lambda: ChatGroq(api_key=self.groq_key, model="llama-3.1-8b-instant"),
        "llama-3.3-70b-versatile": lambda: ChatGroq(api_key=self.groq_key, model="llama-3.3-70b-versatile"),
        "meta-llama/llama-guard-4-12b": lambda: ChatGroq(api_key=self.groq_key, model="meta-llama/llama-guard-4-12b")
    }
    return llm_options[llm_choice]()
```

---

## 🧰 Tech Stack

| Layer          | Technology               | Purpose                      |
| -------------- | ------------------------ | ---------------------------- |
| Backend        | **FastAPI**              | Core API orchestration       |
| Frontend       | **Streamlit**            | Lightweight, easy-to-use UI  |
| Agentic System | **CrewAI**               | Agent/task orchestration     |
| Models         | **Groq-hosted LLaMA**    | Reasoning and generation     |
| RAG            | **EmbedChain + FAISS**   | Contextual retrieval         |
| OCR            | **EasyOCR / pdfplumber** | Document text extraction     |
| Web Search     | **Serper / Firecrawl**   | Live information retrieval   |
| Voice          | **Whisper (planned)**    | Speech-to-text transcription |
| Summarizer     | **Groq small model**     | Context condensation         |

---

## 🔄 Example Flow

```
User enters query / uploads file
      ↓
Streamlit sends request to FastAPI
      ↓
FastAPI routes request to CrewAI (agents + tasks)
      ↓
Agents use tools (Serper, Firecrawl, RAG)
      ↓
Groq model generates final answer
      ↓
FastAPI returns structured JSON response
      ↓
Streamlit displays it with context tracking
```

---

## 🧩 Roadmap

### ✅ MVP (Current)

* [x] Chat system with LLM context
* [x] Multi-agent CrewAI pipeline
* [x] RAG for PDFs and images
* [x] Model selection system
* [x] Summarizer for long context

### 🔜 Next (Upcoming)

* [ ] Persistent long-term memory (SQLite / Supabase)
* [ ] Authentication and user-based sessions
* [ ] Enhanced UI (chat bubbles, markdown, themes)
* [ ] Deployment on Render or GCP Run

---

## ⚡ How to Run (MVP)

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/agentic-wrapper.git
   cd agentic-wrapper
   ```

2. **Install backend dependencies**

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Run FastAPI server**

   ```bash
   uvicorn app:app --reload
   ```

4. **Run Streamlit frontend**

   ```bash
   cd ../frontend
   streamlit run streamlit_app.py
   ```

5. **Open your browser**

   ```
   http://localhost:8501
   ```

---

## 💡 Vision

> The aim is to show that **building AI systems from scratch** — with your own logic, architecture, and design — is not only possible but powerful.

Agentic Wrapper is not meant to compete with commercial AI systems.
Instead, it’s a **demonstration of end-to-end AI engineering** — combining reasoning, retrieval, perception, and orchestration — in one accessible framework.

---

## 🧾 License

MIT License © 2025 – Built and maintained by Dushyant Atalkar

---

## 🤝 Contributing

This MVP is still evolving.
If you’d like to improve the architecture, add frontend components, or contribute agents/tools — feel free to fork and make a pull request!

---

## 🌟 Acknowledgments

* [CrewAI](https://github.com/joaomdmoura/crewAI)
* [Groq](https://groq.com)
* [EmbedChain](https://github.com/embedchain/embedchain)
* [Streamlit](https://streamlit.io)
* [FastAPI](https://fastapi.tiangolo.com)

---

### 🧠 “Build it. Don’t just use it.”

Agentic Wrapper is a proof that **real AI systems** can be built bottom-up —
not just prompted, but *engineered*.




