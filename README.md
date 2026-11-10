# TradeGuard — CTM Autonomous Accrual & Audit Swarm

**An Enterprise Multi-Agent AI System for National Convenience Retail Customer Trade Marketing (CTM) Accounting, Contract RAG, and Chart of Authority Governance**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ctm-tradeguard-swarm.streamlit.app/)

> **Live Interactive Prototype:** **[TradeGuard — CTM Autonomous Swarm (Streamlit Cloud)](https://ctm-tradeguard-swarm.streamlit.app/)**  
> *Explore the 26-Element Master Accrual Ledger (Table B) & Quarterly Schedules (Table A), run the 3-Agent LangGraph Swarm with interactive Human-in-the-Loop (HITL) approvals, test custom claims in the sandbox, and query ChromaDB contracts live in your browser.*

---

## 1. Executive Overview & Business Problem

In National Convenience Retail CTM Accounting, corporate accountants manage millions of dollars in promotional trade funds, quarterly bottle/can case volume rebates, coupon clearinghouse redemptions, and quarterly accrual rate true-ups across major convenience chains (**Hess, BP North America, Wawa, Circle K, and 7-Eleven**).

Auditing an incoming sales executive expense reimbursement or retailer quarterly volume rebate requires cross-referencing **three distinct data sources simultaneously**:
1. **Structured ERP & Accrual Ledger Data (`SQLite`):** Checking whether the customer's contract is `SIGNED` or `UNSIGNED`, verifying sufficient remaining accrual balance (Table B: `accrual_elements_ledger`), and reconciling claimed case volume against actual ERP shipped cases and post-review effective rates (Table A: `element_quarterly_schedule`).
2. **Unstructured Customer Contracts (`ChromaDB Vector RAG`):** Reading customer-specific contract clauses (`data/contracts/*.md`) to catch prohibited items (e.g., store manager gift cards), regional banner exclusions (e.g., West Coast *ampm* vs. Midwest *BP Connect*), tiered case volume thresholds, prior-year YoY growth baselines, and strict quarter-end shipment cutoffs.
3. **Corporate Governance & Chart of Authority (`HITL Routing`):** Enforcing the **6-Point CTM Payment Checklist** and routing verified payouts to the required approver (`CTM Accountant <= $2,500`, `CTM Manager $2,500.01–$50,000`, `Director of Trade Finance $50,000.01–$100,000`, `VP of Finance > $100,000`, plus the mandatory **June 30 Unsigned Contract Escalation Rule**).

**TradeGuard** automates this entire audit lifecycle using a **4-Node LangGraph Multi-Agent Swarm** equipped with SQLite tool calling, header-aware ChromaDB RAG, strict Pydantic structured outputs, automated Golden Dataset benchmarking, a FastAPI microservice, and an interactive Streamlit command center.

---

## 2. Multi-Agent LangGraph Architecture

```text
                  [Incoming CTM Claim / Rebate Request]
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  Node 1: ERP & Accrual Ledger Agent (SQLite Tools)    │
        │  - Tools ONLY: get_element_ledger (Table B),          │
        │                get_quarterly_schedule (Table A)       │
        │  - Writes findings to state["erp_findings"]           │
        └───────────────────────────┬───────────────────────────┘
                                    │ (A2A Context Handoff)
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  Node 2: Contract & Policy Agent (ChromaDB RAG Tool)  │
        │  - Tool ONLY: search_customer_contract (Top-K=3)      │
        │  - Reads claim + state["erp_findings"]                │
        │  - Writes clauses to state["contract_findings"]       │
        └───────────────────────────┬───────────────────────────┘
                                    │ (A2A Context Handoff)
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  Node 3: Financial Auditor & COA Supervisor Agent     │
        │  - Synthesizes Agent 1 + Agent 2 reports              │
        │  - Evaluates 6-Point CTM Checklist & COA Rules        │
        │  - Outputs Pydantic AuditDecision schema              │
        └───────────────────────────┬───────────────────────────┘
                                    │
                      (Conditional Edge: route_after_audit)
                      ┌─────────────┴─────────────┐
                      │                           │
         If APPROVE or REJECT               If ESCALATE
        (<= $2,500 or Failed)        (> $2,500 or Unsigned Contract)
                      │                           │
                      ▼                           ▼
                    [END]         ┌───────────────────────────────┐
                                  │ Node 4: HITL COA Approval Gate│
                                  │ - Pauses workflow for human   │
                                  │   sign-off (Manager /         │
                                  │   Director / VP of Finance)   │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                                [END]
```

### Key Architectural Highlights
* **Least-Privilege Tool Isolation:** Each specialist agent only receives the tools required for its domain (`erp_agent` only accesses SQLite; `contract_agent` only accesses ChromaDB; `auditor_agent` has zero database tools and strictly synthesizes evidence into the Pydantic `AuditDecision` schema).
* **Header-Aware Contract Chunking (`MarkdownHeaderTextSplitter`):** Splits the 5 customer agreements and governance policy by `#`, `##`, and `###` headers (`strip_headers=False`) into **43 self-contained chunks**, ensuring every CTM Element's rules stay intact during vector retrieval.
* **Two-Table Relational Accrual Design (`data/ctm_ledger.db`):**
  * **Table A (`element_quarterly_schedule` — 76 rows):** Tracks quarterly planned vs. actual shipped bottle/can cases (fountain excluded), base rate, effective quarterly accrual rate, accrued dollars, and bottler review notes.
  * **Table B (`accrual_elements_ledger` — 26 elements):** Tracks YTD actual cases, Q1–Q4 accrual rates, YTD accrued budget, YTD spend, remaining balance, and contract signature status across all 5 convenience chains.

---

## 3. Golden Dataset Evaluation Benchmark (`15/15 = 100.0%`)

The swarm was benchmarked against a 15-claim Golden Dataset (`data/claims/sample_claims.json`) spanning all 5 convenience retail chains and every major CTM accounting discrepancy type (with ground-truth labels stripped prior to agent invocation):

| Metric | Score | Details |
| :--- | :--- | :--- |
| **Decision Accuracy** | **100.0% (15/15)** | Exact match across `APPROVE`, `REJECT`, and `ESCALATE` |
| **COA Approver Routing Accuracy** | **100.0% (15/15)** | Exact match across `CTM Accountant`, `CTM Manager`, `Director of Trade Finance`, `VP of Finance`, and `None` |
| **Full Exact-Match Accuracy** | **100.0% (15/15)** | Both Decision and COA Approver matched on every claim |
| **Average Audit Latency** | **13.56s / claim** | End-to-end 3-agent execution (SQLite + ChromaDB + COA Auditor) |

### Sample Discrepancies Caught by the Swarm
* **`CLM-HESS-004` (\$4,500 Store Manager Gift Cards):** Rejected via ChromaDB RAG (`HESS-SUMMER` explicitly prohibits gift cards/cash equivalents despite sufficient SQLite balance).
* **`CLM-BP-001` (\$32,900 12-Pack Rebate):** Rejected via A2A teamwork—SQLite reported `94,000` Q3 actual cases while ChromaDB reported that the claimed \$0.35/case Tier 2 rate requires `100,000` cases.
* **`CLM-BP-002` (\$6,400 Midwest *BP Connect* Combo):** Rejected via ChromaDB RAG (`BP-COMBO` is restricted strictly to West Coast *ampm* stores).
* **`CLM-WAWA-001` (\$1,950 JBP Summit):** Escalated to `Director of Trade Finance`—caught that Wawa's contract is `UNSIGNED` past June 30, overriding the normal $\le \$2,500$ CTM Accountant approval tier.
* **`CLM-WAWA-002` (\$90,500 Cold Vault Rebate):** Rejected for including 4,500 cases delivered on Oct 2 (Q4) in violation of the strict Q3 shipment cutoff (`176,500` ERP cases vs. `181,000` claimed).
* **`CLM-CK-002` (\$54,300 YoY Growth Incentive):** Rejected after calculating that `362,000` Q3 cases vs. `350,000` prior-year baseline (`+3.43%` growth) only qualifies for Tier 1 (\$0.10/case), not Tier 2 (\$0.15/case).

---

## 4. Progressive 6-Stage Repository Structure

This project is organized into progressive stages—each building on the previous stage to demonstrate clean, incremental AI engineering:

```text
CustomerTradeMarketing_CR/
├── data/
│   ├── contracts/                  # 5 Unstructured Markdown Customer Contracts (26 Elements)
│   ├── governance/                 # CTM Governance Policy (6-Point Checklist & 4-Tier COA)
│   ├── claims/sample_claims.json   # 15-Claim Golden Evaluation Dataset
│   ├── seed_data.json              # Raw ERP & Ledger fixture rows (76 Qtr rows, 26 Elements)
│   ├── ctm_ledger.db               # SQLite Database (Table A & Table B)
│   ├── chroma_db/                  # Persistent ChromaDB Vector Store (43 chunks)
│   └── eval_results.json           # Full 15/15 Benchmark Output & AI Rationales
├── Phase1_SingleCTMAgent/          # Stage 1: SQLite Seeding + Single Tool-Calling Agent
├── Phase2_RAGChroma/               # Stage 2: Header-Aware Contract Ingestion + ChromaDB RAG Tool
├── Phase3_MultiAgentSwarm/         # Stage 3: 4-Node LangGraph Swarm + Interactive Terminal HITL Gate
├── Phase4_EvalObservability/       # Stage 4: Automated 15-Claim Benchmark Runner & LangSmith Tracing
├── Phase5_FastAPIDashboard/        # Stage 5: FastAPI REST Microservice + Web Command Center
└── Phase6_StreamlitDashboard/      # Stage 6: Interactive End-User Streamlit Prototype & Sandbox
```

---

## 5. Tech Stack

* **Package & Environment Manager:** `uv` (`pyproject.toml`, `uv.lock`)
* **LLM & Embeddings:** OpenAI `gpt-4o-mini` & `text-embedding-3-small` (`langchain-openai`)
* **Agent & Graph Orchestration:** `LangChain` + `LangGraph` (`StateGraph`, Conditional Edges, HITL Gates)
* **Structured Validation:** `Pydantic` (`BaseModel`, `Literal`, `Field`)
* **Databases:** `SQLite3` (Relational ERP & Accrual Ledger) + `ChromaDB` (Local Vector Store)
* **Observability & Evaluation:** `LangSmith` Distributed Tracing + Custom Golden Dataset Evaluator
* **APIs & Interactive UI:** `FastAPI` + `Uvicorn` + `Streamlit` + `Pandas`

---

## 6. Quickstart & Running Locally

### 1. Clone & Install Dependencies (`uv`)
```powershell
uv sync
```
Create a `.env` file at the project root with your OpenAI API key:
```ini
OPENAI_API_KEY=sk-proj-your-key-here
```

### 2. Seed SQLite & Index Contracts into ChromaDB
```powershell
uv run Phase1_SingleCTMAgent/seed_database.py
uv run Phase2_RAGChroma/ingest_contracts.py
```

### 3. Run the Interactive Multi-Agent CLI Swarm (`Phase 3`) or Evaluation Benchmark (`Phase 4`)
```powershell
uv run Phase3_MultiAgentSwarm/swarm.py
uv run Phase4_EvalObservability/evaluate_swarm.py
```

### 4. Launch the FastAPI Server (`Phase 5`) or Streamlit Prototype (`Phase 6`)
```powershell
# Option A: FastAPI REST Backend + Web Dashboard (http://127.0.0.1:8000)
cd Phase5_FastAPIDashboard
uv run uvicorn app:app --reload

# Option B: Interactive 4-Tab Streamlit Prototype (http://localhost:8501)
cd Phase6_StreamlitDashboard
uv run python -m streamlit run streamlit_app.py
```
