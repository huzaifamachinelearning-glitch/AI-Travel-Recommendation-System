# AI Travel Advisor 🌏

> RAG-powered travel recommendation system for India — production-ready, deployable, interview-worthy.

[![CI/CD](https://github.com/huzaifamachinelearning-glitch/AI-Travel-Advisor/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/huzaifamachinelearning-glitch/AI-Travel-Advisor/actions)

---

## What this does differently

Most travel recommendation projects use basic ML classification. This uses **RAG (Retrieval Augmented Generation)**:

| Old Approach | This Project |
|---|---|
| Predict if rating will be high/low | Retrieve relevant destinations + generate personalized advice |
| Synthetic random data | Structured knowledge base with real destination facts |
| Groq gets 5 city names | Groq gets full destination data (budget, food, transport, tips) |
| `python app.py` to run | Docker container + Railway deployment + GitHub Actions CI/CD |

---

## Architecture

```
User Request
    ↓
FastAPI Backend
    ↓
RAG Pipeline:
  1. Retrieve → Score destinations by user preferences
  2. Augment  → Build rich context (budget, food, transport, safety)
  3. Generate → Groq LLaMA 3.3 creates personalized advice
    ↓
Structured JSON Response (Pydantic validated)
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | FastAPI | Async, auto-docs, production-standard |
| LLM | Groq + LLaMA 3.3 70B | Fast inference, free tier available |
| RAG | Custom retrieval + knowledge base | No external vector DB needed to start |
| Validation | Pydantic v2 | Type-safe API contracts |
| Deploy | Railway + Docker | Simple, affordable, professional |
| CI/CD | GitHub Actions | Auto-deploy on push |

---

## Quick Start

### 1. Clone
```bash
git clone https://github.com/huzaifamachinelearning-glitch/AI-Travel-Advisor.git
cd AI-Travel-Advisor/backend
```

### 2. Setup environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # Fill in your GROQ_API_KEY
```

### 3. Run locally
```bash
uvicorn main:app --reload
# Visit: http://localhost:8000/docs (auto-generated API docs!)
```

### 4. Run with Docker
```bash
docker build -t ai-travel-advisor .
docker run -p 8000:8000 --env-file .env ai-travel-advisor
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check (used by Railway) |
| POST | `/api/v1/recommend` | Get personalized recommendations |
| POST | `/api/v1/chat` | Conversational travel assistant |
| GET | `/api/v1/destinations` | Browse all destinations |
| GET | `/api/v1/destinations/{id}` | Destination details |
| GET | `/docs` | Interactive API documentation |

### Example Request
```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "budget_per_day": 3,
    "duration_days": 5,
    "companions": "Friends",
    "interests": ["Adventure", "Nature"],
    "season": "Winter",
    "min_safety": 3.5
  }'
```

---

## Project Structure

```
ai-travel-advisor/
├── backend/
│   ├── main.py                 # FastAPI app entry point
│   ├── Dockerfile              # Container definition
│   ├── requirements.txt        # Pinned dependencies
│   ├── .env.example            # Environment template (commit this)
│   ├── routers/
│   │   ├── recommend.py        # POST /recommend
│   │   ├── chat.py             # POST /chat
│   │   └── destinations.py     # GET /destinations
│   ├── services/
│   │   ├── rag_service.py      # Retrieval + context building
│   │   └── llm_service.py      # Groq API wrapper
│   └── models/
│       └── schemas.py          # Pydantic request/response models
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # Auto-deploy on push to main
└── .gitignore
```

---

## Deployment (Railway)

1. Push code to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Add environment variable: `GROQ_API_KEY`
4. Railway detects Dockerfile automatically
5. Every `git push main` → auto-deploys via GitHub Actions

**Keep alive (prevent sleep mode):** Use [UptimeRobot](https://uptimerobot.com) to ping `/health` every 5 minutes on free tier.

---

## Git Workflow (professional)

```bash
# Feature branch workflow
git checkout -b feature/add-state-filter
# make changes
git add .
git commit -m "feat: add state-based destination filtering"
git push origin feature/add-state-filter
# Open Pull Request on GitHub → merge to main → auto-deploy
```

---

## Future Upgrades

- [ ] ChromaDB for real vector embeddings
- [ ] React frontend with map visualization
- [ ] User authentication (JWT)
- [ ] Trip itinerary PDF generation
- [ ] Real data from Google Places API

---

## Author

**Mohammad Huzaifa** — [@huzaifamachinelearning-glitch](https://github.com/huzaifamachinelearning-glitch)