import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from schemas import AuditDecision
from tools import get_element_ledger, get_quarterly_schedule, search_customer_contract

#load open ai key from .env file 
load_dotenv()

'''
system_prompt given to LLM before it sees claim. Tells agent who it is (CTM accountant), 
what tools to call in what order and how to map the results to the checklist and COA.

get_element_ledger and get_quarterly_schedule tools against SQLite and formats its final answer into 
Pydantic AuditDecision (result["structured_response"])

audit_claim(): takes a single claim dictionary (from sample.json) and returns an AuditDecision object. 
    initialize gpt-40-mini with temperature=0 (max determinism - not creative guessing/randommness)
    bundles 2 SQLite tools into a list so agent knows what tools are in its toolbox. 
    create_react_agent(): create a ReAct (Reason + Act), builds the ReAct loop:
        -agent reads system_prompt and users claim.
        -agent decides which tool(s) to call.
        -LangGraph executes function against data/ctm_ledger.db and feeds the JSON rows back to the LLM. 
         once the LLM has all the fact, response_format=AuditDecision forces it to populate the Pydantic schema. 
        -when you pass response_format=AuditDecision to create_reacact_agent(), LangGraph stores the 
        validated Pydantic AuditDecison instance inside result["structured_response"]
        
main(): loads the 14 stest claims from data/claims/sample_claims.json tion a Python list of dicts. 
    claims[:3] uses list slicing to grab just the first 3 claims to test fast w/o running all 15 at once. 
    formats and prints audit results 
    (${claim['amount_requested']:,.2f} formats floats with commas and 2 decimal places like $1,000.00)

Agent loop:     
When agent.py audits one claim (say CLM-Hess-003, $54k Cold Vault rebate claim), gpt-40-mini is called 
   multiple times in a loop. 
      1. agent.py sends the system_prompt + claims json + tool list to OpenAI gpt-40-mini
      2. LLM reasons #1: "I need to check Table B for balance AND table A for 2026-Q3 actual cases"
      3. tools.py (local SQLite) LLM asks to run BOTH tools
         -Runs get_element_ledger('Hess-20oz') -> returns $48k balance
         -Runs get_quarterly_schedule('Hess-20z', '2026-Q3') -> returns 120k actual cases
         -Then sends raw SQLite JSON rows back to OpenAI gpt-40-mini
      4. LLM reasons#2: "Compare claim vs SQLite":
            -$54k requsted >$48k balance ->Item2 =False   
            -$135k caes claimed > 120k actual cases -> Item5 = False
            -Per COA rule, since Item2 & Item5 failed -> Reject ($0.00)
      5. schemas.py (AduitDecision): LLM fills out the Pydantic form + writes Rationale 
 
clean_claim: dictionary comprehension using k/v pairs that compacts a for loop with an if fiter that copies a dict while skipping certain keys.
The dict comprehension writes 4 lines of code into a single expression. It is equivalent to: 
  clean_claim = {
    k: v                                                     # 3. Put key (k) and value (v) into the new dictionary
    for k, v in claim.items()                                # 1. loop/grab each (key, value) pair from the original claim
    if k not in ("expected_outcome", "expected_approver")    # 2. filter/only keep it if the key is NOT one of these two.
}  

Example...
claim.items():Takes a dictionary from sample_claims.json and unpacks it into (key, value) pairs, one at a time:
    Pair 1: k = "claim_id", v = "CLM-HESS-001"
    Pair 2: k = "amount_requested", v = 1850.0
    ...
    Pair 12: k = "expected_outcome", v = "APPROVE"
    Pair 13: k = "expected_approver", v = "CTM Accountant"

for k, v in claim.items():
Loops through every one of those (key, value) pairs, storing the key name in k and the value in v.
if k not in ("expected_outcome", "expected_approver"):Checks the key name k:
When k is "claim_id" or "amount_requested", is it in that tuple? No (not in is True) 
so Python keeps it. When k reaches "expected_outcome" or "expected_approver", is it in that tuple? 
Yes (not in is False), Python excludes.

{ k: v ... }:Builds the brand-new dictionary (clean_claim) containing only the pairs that passed the if test.

'''

#CLAIMS_PATH = Path("data/claims/sample_claims.json")
ROOT_DIR = Path(__file__).resolve().parent.parent
CLAIMS_PATH = ROOT_DIR / "data/claims/sample_claims.json"

SYSTEM_PROMPT = """You are an expert Customer Trade Marketing (CTM) Corporate Accountant and Financial Auditor.
Your job is to audit incoming CTM promotional expense claims and quarterly volume rebate claims using the
mandatory CTM Payment Checklist and Chart of Authority (COA).
Always follow these steps:
1. Call 'get_element_ledger' using the claim's element_id to check:
    - Checklist Item#1: Is 'contract_status' SIGNED? (If UNSIGNED after June 30, Item#1 is False and ALL
      valid claims require mandatory escalation to 'Director of Trade Finance', regardless of dollar amount.)
    - Checklist Item#2: Is 'remaining_balance >= amount_requested'? (Note: if 'element_id' is an upfront
      advance element with a negative remaining balance, no mid-year claims are allowed.)
2. If the claim is for a quarterly volume rebate or references case volume, call 'get_quarterly_schedule'
   for that 'element_id' and 'fiscal_quarter' to check:
    - Checklist Item#5: Does 'claimed_cases' match 'actual_cases' in ERP?
      Does 'claimed_rate' match the earned rate for that quarter? (If a claim is submitted before the
      quarter closes with no schedule row in ERP, reject it.)
3. Call 'search_customer_contract' using the customer name and element_id
   (e.g., "BP BP-COMBO eligible banners rules") to read the actual contract clauses and check:
    - Checklist Item#3: Is 'receipt_or_pop_attached' TRUE?
      (For coupon claims, math must be strictly on actual redeemed coupons, never printed coupons.)
    - Checklist Item#4: Is the expense eligible under the specific element's contract rules?
      Watch out for prohibited items (like gift cards), regional/banner restrictions (like ampm vs BP Connect),
      tiered volume thresholds, or YoY baseline requirements.
4. Apply the Chart of Authority (Checklist Item#6):
    - If ANY of Checklist items #2, #3, #4 or #5 is False -> decision = 'REJECT', required_approver = 'None',
      verified_payout_amount = 0.0
    - If Items #2-#5 are True but Item#1 is False (UNSIGNED contract past June 30) -> decision = 'ESCALATE',
      required_approver = 'Director of Trade Finance'.
    - If all Items #1-#5 are True:
      * Up to $2,500.00 -> decision = 'APPROVE', required_approver = 'CTM Accountant'
      * $2,500.01 to $50,000.00 -> decision = 'ESCALATE', required_approver = 'CTM Manager'
      * $50,000.01 to $100,000.00 -> decision = 'ESCALATE', required_approver = 'Director of Trade Finance'
      * Over $100,000.00 -> decision = 'ESCALATE', required_approver = 'VP of Finance'
"""
def audit_claim(claim:dict) -> AuditDecision:
    "Run the CTM Auditor agent on a single claim dictionary"
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_element_ledger, get_quarterly_schedule, search_customer_contract]
    
    agent = create_react_agent(
        model = llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        response_format = AuditDecision,
    )

    """json.dumps() is passed the entire claim dictionary, including 'expected outcome' and 'expected approver'. """
    """This is a data leakage issue, because the agent can read the expected outcome and cheat."""
    #user_message = f"Please audit the following CTM claim: \n{json.dumps(claim, indent=2)}" 
    
    "Remove expected_outcome and expected_approver from the claim dictionary so the agent doesn't cheat and see the answer in advance."
    clean_claim = {
        k: v for k, v in claim.items() if k not in ("expected_outcome", "expected_approver")
    }
    user_message = f"Please audit the following CTM claim:\n{json.dumps(clean_claim, indent=2)}"
    
    result = agent.invoke({"messages": [("user", user_message)]})
    
    return result['structured_response']

def main() -> None:
    with open(CLAIMS_PATH, 'r', encoding="utf-8") as f:
        claims = json.load(f)
        
    #Test first 3  Hess claims (Clean, Missing Receipt, Volume overclaim)
    for claim in claims[3:8]: #testing 5 claims specifically require reading gift cards, couponts, tiered volume, banner exclusion, unsigned contract
        print(f"\n{'='*60}")
        print(f"Auditing {claim['claim_id']} ({claim['customer_name']} - {claim['element_id']})")
        print(f"Requested: ${claim['amount_requested']:,.2f} | Expected: {claim['expected_outcome']}")
        
        decision = audit_claim(claim)
        
        print(f"\n-> AGENT DECISION: {decision.decision} (Approver: {decision.required_approver})")
        print(f"-> Verified Payout: ${decision.verified_payout_amount:,.2f}")
        print(f"-> Checklist: [1.Contract:{decision.item_1_valid_contract} | 2.Funds:{decision.item_2_sufficient_funds} | 3.Receipt/POP:{decision.item_3_receipt_or_pop_attached} | 4.Eligible:{decision.item_4_expense_eligible} | 5.Math/Vol:{decision.item_5_math_and_volume_verified}]")
        print(f"-> Rationale: {decision.rationale}")
if __name__ == "__main__":
    main()