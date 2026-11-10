import json
import time
from pathlib import Path
from dotenv import load_dotenv

from swarm import build_swarm_graph

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
CLAIMS_PATH = ROOT_DIR / "data/claims/sample_claims.json"
RESULTS_PATH = ROOT_DIR / "data/eval_results.json"


'''
run_evaluation():
    (for idx, claim in enumerate(claims, start=1):):
    enumerate(..., start=1) loops through all 15 claims while automatically 
    counting idx = 1, 2, 3 ... 15 so we can print [01/15], [02/15], etc.
    
    (time.perf_counter()):
    Starts a high-precision stopwatch right before swarm.invoke(...) and subtracts start_time 
    right after it finishes, giving the exact number of seconds the 3 agents took to audit that claim.
    
    (is_full_match = is_decision_correct and is_approver_correct):
    A claim only gets [PASS] if both the decision (APPROVE / REJECT / ESCALATE) and the 
    Chart of Authority approver (CTM Accountant / CTM Manager / Director of Trade Finance / VP of Finance / None)
    match the ground truth!

    (f"... {expected_decision:<8} ..."):
    In Python f-strings, :<8 and :<25 left-align and pad strings to 8 or 25 characters wide so 
    all 15 rows line up in a neat table in terminal!
    
    The clean_claim dict excludes 'expected_outcome' and 'expected_approver' so AI swarm does not see them. 
    The original claim dict includes 'expected_outcome' and 'expected_approver'. 
    expected_decision = claim['expected_outcome']: Grabs "ESCALATE" directly from claim.
    expected_approver = extract_expected_role(claim['expected_approver'])):
        Passes just this 1 claim's 'expected_approver' string ("Director of Trade Finance (Mandatory June 30...)") 
        up to extract_expected_role()
        Then extract_expected_role() chops off the parenthetical explanation and returns "Director of Trade Finance", storing it in expected_approver.
        Since the for loop runs 15 times, extract_expected_role() gets called 15 times total—once per claim. 
        

    Running the Swarm with a Stopwatch (time.perf_counter())
    start_time = time.perf_counter()): Clicks "Start" on Python's high-precision stopwatch right before calling the AI (e.g., 1000.00 seconds).
    final_state = swarm.invoke({"claim": clean_claim, "auto_hitl": True})):
        Runs your entire 4-node LangGraph workflow (erp_agent -> contract_agent -> auditor_agent  -> hitl_agent) on clean_claim.
        Passing "auto_hitl": True tells Node 4 not to pause for keyboard input.
        When the workflow finishes, it returns the completed SwarmState clipboard into final_state.
        
    latency = time.perf_counter() - start_time): Clicks "Stop" on the stopwatch and subtracts start_time (1014.01 - 1000.00 = 14.01 seconds). 
    That's how long this 1 claim took!
    
    total_latency += latency: Adds this claim's 14.01 seconds onto running total (total_latency) to calculate the average latency 
    across all 15 claims at the end.
    
    Unpacking what AI Swarm returns in final_state:
        actual_decision_obj = final_state["audit_decision"]
            Pulls the Pydantic AuditDecision object (filled out by Node 3, the Financial Auditor Agent) out of the final_state dictionary.
        
    actual_decision = actual_decision_obj.decision
        actual_decision: Reads what the AI picked for decision (e.g., "ESCALATE").
    
    actual_approver = actual_decision_obj.required_approver
        actual_approver: Reads what the AI picked for required_approver (e.g., "Director of Trade Finance").
        
        
    Grading AI (Actual vs Expected) & Updating Scoreboard
    is_decision_correct = actual_decision == expected_decision:
        Compares the AI's decision ("ESCALATE") to the answer key ("ESCALATE").  A match means is_decision_correct becomes True.
        
    is_approver_correct = actual_approver == expected_approver:
        Compares AI's approver ("Director of Trade Finance") to answer key ("Director of Trade Finance"). A match means is_approver_correct becomes True.
    
    
    is_full_match = is_decision_correct and is_approver_correct:
        Python's and operator returns True only if both is_decision_correct AND is_approver_correct are True. 
        If the AI got "ESCALATE" right, but picked "CTM Manager" instead of "Director of Trade Finance", is_full_match would be False. 
        
    Before the loop started, set three counters (decision_matches, approver_matches, exact_full_matches) to 0.
    Each time a claim passes one of those if checks, += 1 adds 1 point to that counter (until all three counters reached 15 out of 15.).
    
'''


def extract_expected_role(expected_approver_str: str) -> str:
    """Extracts the clean approver title before any parenthetical note. 
       Example: 'Director of Trade Finance (Mandatory June 30...)' -> 'Director of Trade Finance'
    """
    return expected_approver_str.split("(")[0].strip()


def run_evaluation() ->None:
    with open(CLAIMS_PATH, 'r', encoding='utf-8') as f:
        claims = json.load(f)
        
    swarm = build_swarm_graph()
    
    total_claims = len(claims)
    decision_matches = 0
    approver_matches = 0
    exact_full_matches = 0
    total_latency = 0
    detailed_results = []
    
    print(f"\n{'='*85}")
    print(f"RUNNING CTM SWARM GOLDEN DATASET EVALUATION ({total_claims} CLAIMS)")
    print(f"{'='*85}")
    
    for idx, claim in enumerate(claims, start=1):
        """Start ground-truth answer key so swarm cannot peak"""
        clean_claim = {
            k: v for k, v in claim.items() if k not in ("expected_outcome", "expected_approver")
        }
        
        expected_decision = claim['expected_outcome']
        expected_approver = extract_expected_role(claim['expected_approver'])
        
        """Time multi-agent swarm exectuion with auto_hitl=True"""
        start_time = time.perf_counter()
        final_state = swarm.invoke({"claim": clean_claim, "auto_hitl": True})
        latency = time.perf_counter() - start_time
        total_latency += latency
        
        
        actual_decision_obj = final_state["audit_decision"]
        actual_decision = actual_decision_obj.decision
        actual_approver = actual_decision_obj.required_approver
        
        """Graph Decision & COA Approver against Ground Truth"""
        is_decision_correct = actual_decision == expected_decision
        is_approver_correct = actual_approver == expected_approver
        is_full_match = is_decision_correct and is_approver_correct
        
        if is_decision_correct:
            decision_matches +=1
        if is_approver_correct:
            approver_matches +=1
        if is_full_match:
            exact_full_matches +=1
            
        status_icon = 'PASS' if is_full_match else 'FAIL'
        print(
            f"[{idx:02d}/{total_claims:02d}] {claim['claim_id']} ({claim['element_id']:<13}) | "
            f"Expected: {expected_decision:<8} ({expected_approver:<25}) | "
            f"Actual: {actual_decision:<8} ({actual_approver:<25}) | "
            f"{latency:5.2f}s | [{status_icon}]"
        )
        
        detailed_results.append(
            {
                "claim_id": claim["claim_id"],
                "customer_name": claim["customer_name"],
                "element_id": claim["element_id"],
                "amount_requested": claim["amount_requested"],
                "expected_decision": expected_decision,
                "actual_decision": actual_decision,
                "expected_approver": expected_approver,
                "actual_approver": actual_approver,
                "is_full_match": is_full_match,
                "verified_payout_amount": actual_decision_obj.verified_payout_amount,
                "latency_seconds": round(latency, 2),
                "rationale": actual_decision_obj.rationale,
            }
        )
        
    #Compute summary metrics
    summary = {
        "total_claims": total_claims,
        "decision_accuracy_pct": round((decision_matches / total_claims) * 100, 1),
        "approver_accuracy_pct": round((approver_matches / total_claims) * 100, 1),
        "exact_match_accuracy_pct": round((exact_full_matches / total_claims) * 100, 1),
        "average_latency_seconds": round(total_latency / total_claims, 2),
        "total_latency_seconds": round(total_latency, 2),
        "results": detailed_results,
    }
        
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n{'='*85}")
    print("GOLDEN DATASET EVALUATION SCORECARD")
    print(f"{'='*85}")
    print(f"Total Claims Audited:      {total_claims}")
    print(f"Decision Accuracy:         {decision_matches}/{total_claims} ({summary['decision_accuracy_pct']}%)")
    print(f"COA Approver Accuracy:     {approver_matches}/{total_claims} ({summary['approver_accuracy_pct']}%)")
    print(f"Full Exact-Match Accuracy: {exact_full_matches}/{total_claims} ({summary['exact_match_accuracy_pct']}%)")
    print(f"Average Latency per Claim: {summary['average_latency_seconds']}s (Total: {summary['total_latency_seconds']}s)")
    print(f"Saved Detailed Report To:  {RESULTS_PATH}")
    print(f"{'='*85}\n")


if __name__ == "__main__":
    run_evaluation()