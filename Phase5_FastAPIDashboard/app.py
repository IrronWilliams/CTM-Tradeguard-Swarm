import json
import sqlite3
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from swarm import build_swarm_graph


# Paths to shared root data files and local dashboard HTML
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
DB_PATH = ROOT_DIR / "data/ctm_ledger.db"
CLAIMS_PATH = ROOT_DIR / "data/claims/sample_claims.json"
EVAL_RESULTS_PATH = ROOT_DIR / "data/eval_results.json"
DASHBOARD_HTML_PATH = CURRENT_DIR / "dashboard.html"

'''
The current paths are relative to the app.py file. Points to where the respective files live. 

@app.get('/')
When open http://127.0.0.1:8000/ in browser, FileResponse(DASHBOARD_HTML_PATH) 
sends dashboard.html straight to your browser. 

@app.get('/api/ledger')
Connects to SQLite (data/ctm_ledger.db) and runs SELECT * on both 
TableB (accrual_elements_ledger) and TableA (element_quarterly_schedule).

[dict(row) for row in cursor.fetchall()] list comprehension that converts all 
TableB rows and TableA rows into dictionaries so FastAPI can send 
them to the dashboard as JSON.

@app.get('/api/eval-results')
Reads data/eval_results.json (15/15 benchmark scorecard and rationales) 
and data/claims/sample_claims.json and sends them to the dashboard 
on page load.


@app.post('/api/audit/{claim_id}')
When click "▶ Run Live Swarm" on any claim in the dashboard, the browser 
sends a POST request with that claim_id (e.g., /api/audit/CLM-BP-001).

next((c for c in claims if c["claim_id"] == claim_id), None): 
Searches claims for the matching claim_id 
(or returns None if not found, raising a 404 HTTP error).

Invokes 3-Agent LangGraph Swarm in real time and returns 
erp_findings (Agent 1), contract_findings (Agent 2), 
and decision.model_dump() (which converts Agent 3's 
Pydantic AuditDecision object into a Python dictionary for JSON).

'''



app = FastAPI(title="CTM Autonomous Accrual & Audit Swarm")

#Serve the HTML Dashboard
@app.get('/')
def serve_dashboard():
    """Serve single-page CTM Accural & Audit Dashboard"""
    return FileResponse(DASHBOARD_HTML_PATH)

#Get /api/ledger returns TableB(customer elements) & TableA(quarterly schedule)
@app.get('/api/ledger')
def get_full_ledger() -> dict:
    """Queries SQLite for all customer elements and quarterly schedules"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM accrual_elements_ledger ORDER BY customer_id, element_id")
        elements = [dict(row) for row in cursor.fetchall()]
        
        cursor.execute("SELECT * FROM element_quarterly_schedule ORDER BY element_id, quarter")
        schedules = [dict(row) for row in cursor.fetchall()]
    return {"elements": elements, "schedules": schedules}

#Get /api/eval-results returns 15/15 evaluation scorecard + sample claims
@app.get('/api/eval-results')
def get_evaluation_results() -> dict:
    """Returns golden dataset evaluation results and incoming payloads"""
    with open(CLAIMS_PATH, "r", encoding="utf-8") as f:
        claims = json.load(f)
        
    eval_data ={}
    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    return {"evaluation": eval_data, "claims": claims}
    

#POST /api/audit/{claim_id} returns live 3 agent LangChain Swarm on a specific claim
@app.post('/api/audit/{claim_id}')
def run_live_swarm_audit(claim_id: str) -> dict:
    """Runs live Multi-Agent LangGraph Swarm on a selected claim ID"""
    with open(CLAIMS_PATH, "r", encoding="utf-8") as f:
        claims = json.load(f)
        
    #Fiind matching claim by claim_id
    target_claim = next((c for c in claims if c["claim_id"] == claim_id), None)
    if not target_claim:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found")    
        
    #Strip ground-truth keys before invoking the swarm
    clean_claim = { k: v for k, v in target_claim.items() if k not in ("expected_outcome", "expected_approver")}
    
    
    swarm = build_swarm_graph()
    final_state = swarm.invoke({"claim": clean_claim, "auto_hitl": True})
    decision= final_state["audit_decision"]
    
    return {
        "claim_id": claim_id,
        "erp_findings": final_state["erp_findings"],
        "contract_findings": final_state["contract_findings"],
        "human_approval_status": final_state["human_approval_status"],
        "audit_decision": decision.model_dump(),
    }