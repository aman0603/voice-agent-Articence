# Voice RAG Agent 🤖🎙️

A production-grade Voice AI Agent that provides instant, voice-based answers from technical manuals.

## 🚀 Key Features

- **Zero-Latency**: Uses Browser TTS for instant feedback or Kokoro Neural TTS for high quality.
- **RAG Engine**: Hybrid Search (FAISS + BM25) + Cross-Encoder Reranking for high accuracy.
- **Dual LLM Support**: Defaults to **NVIDIA (Llama 3)**, falls back to **Google Gemini**.
- **Modern Stack**: React 19 Frontend + FastAPI Backend + SSE Streaming.

---

## 🛠️ Quick Start

### 1. Prerequisites
- Python 3.11+
- Node.js 18+
- NVIDIA API Key (Recommended) or Google Gemini API Key.

### 2. Setup Environment
```powershell
# Create virtual environment
uv venv
.venv\Scripts\activate

# Install dependencies
uv sync

# Configure API Keys
cp .env.example .env
# Edit .env and add your NVIDIA_API_KEY or GEMINI_API_KEY
```

### 3. Build Knowledge Base
Download and index the manuals (files are in `data/manuals`):
```powershell
uv run python scripts/build_index.py
```

### 4. Run the System

**Terminal 1: Backend**
```powershell
$env:PYTHONPATH = "."
uv run uvicorn src.clean_server:app --reload --port 8002
```

**Terminal 2: Frontend**
```powershell
cd frontend-react
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 🏗️ Architecture

### Frontend (React)
- **Voice Capture**: Web Speech API for low-latency ASR.
- **Protocol**: SSE (Server-Sent Events) for real-time text & audio streaming.
- **Audio Playback**: 
  - **Browser TTS**: Zero latency (Robotic).
  - **Kokoro TTS**: Neural quality (3s latency on CPU), streamed as base64 audio chunks.

### Backend (FastAPI)
- **Intent Detection**: Classifies queries (Troubleshooting, Config, etc.).
- **Retrieval**: 
  - Dense Search (FAISS)
  - Sparse Search (BM25)
  - Reranker (Cross-Encoder)
- **Generation**:
  - **Primary**: NVIDIA API (Llama 3)
  - **Fallback**: Google Gemini
- **TTS**: Kokoro-82M (Local PyTorch Model).

---

## 📂 Project Structure

```
├── frontend-react/     # React Application
├── src/
│   ├── clean_server.py # Main FastAPI Server
│   ├── config.py       # Configuration
│   ├── retrieval/      # RAG Logic (FAISS/BM25)
│   ├── generation/     # LLM Clients (NVIDIA/Gemini)
│   └── voice/          # Kokoro TTS
└── data/               # PDF Manuals & Index
```

## 📜 License
MIT
