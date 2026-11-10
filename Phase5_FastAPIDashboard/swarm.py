import json
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from schemas import AuditDecision
from tools import get_element_ledger, get_quarterly_schedule, search_customer_contract


load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent.parent
CLAIMS_PATH = ROOT_DIR / "data/claims/sample_claims.json"


'''
SwarmState: TypeDict is a standard Python class that lets you define a dict with a strict set of named keys and value types.
Think of swarmstate as a shared folder that gets passed from agent to agent to human in loop. 

    When the workflow starts, only claim is filled in.
    Node 1 (erp_ledger_agent_node) reads claim and fills in erp_findings.
    Node 2 (contract_policy_agent_node) reads claim + erp_findings and fills in contract_findings.
    Node 3 (financial_auditor_node) reads claim + erp_findings + contract_findings and fills in audit_decision.
    Node 4 (hitl_approval_node) reads audit_decision and fills in human_approval_status

Node Agent 1 ERP & Accrual Ledger Agent
erp_ledger_agent_node(): Only has access to the 2 SQLite tools with no acess to ChromaDB.
    Every LangGraph node function takes state: (shared SwarmState) and returns a dict containing only the new key(s) it wants to update in state. 
    Ensure to add a trailing space at end of each line in prompt to prevent words from sticking together (rulesfrom, rules from, etc.)

    When erp_agent.invoke(...) finishes its ReAct loop, result["messages"] holds the conversation history: 
        [HumanMessage, AIMessage(tool_call), ToolMessage(SQL JSON), AIMessage(Final Summary)].
        Index [-1] grabs the very last item in a list, and .content extracts the plain summary string written by the agent.
        
    (return {"erp_findings": erp_report}): LangGraph takes this returned dictionary and automatically merges erp_findings into SwarmState!


Node Agent 2 Contract & Policy Agent
contract_policy_agent_node(): only has access to ChromaDB vector search tool. 
    the user_msg variable is where the agent-to-agent sharing occures. Node 2 reads state['erp_findings'] written by Node 1 via :
         f"ERP Findings from Agent 1:\n{state['erp_findings']}\n\n"
    The sharing is important b/c if Node1 discoves that BP shipped 94k cases in Q3, Node2 will see this number and knows to check the 94k
    volume tier in the contract.

    result["messages"][-1].content) and writes it into state["contract_findings"] -> this grabls Node2's final summary. 
'''

'''
Node 3 Financial Auditor & COA Supervisor Agent
financial_auditor_node():
    llm.with_structured_output(AuditDecision) instead of create_react_agent:
        Nodes 1 and 2 needed create_react_agent b/c they had to call tools in a loop (SQLite and ChromaDB).
        Node 3 does not need to call any tool. Agents 1 and 2 have already done the legwork and wrote reports to state. 
        
        Calling llm.with_structured_output(AuditDecision) attaches the Pydantic clipboard directly to gpt-4o-mini so 
        it reads state['claim'], state['erp_findings'], and state['contract_findings'] and returns a validated
        AuditDecision object in one single LLM call.
        
        (status = "AUTO_APPROVED" if decision.decision == "APPROVE" else "AUTO_REJECTED"):
        A 1-line ternary expression: if the Auditor's decision is "APPROVE", it sets status to "AUTO_APPROVED"; 
        otherwise it defaults to "AUTO_REJECTED" (and if the decision is "ESCALATE", Node 4 will overwrite human_approval_status 
        with the human's actual sign-off).
        

route_afer_audit(): 
    This is a conditional edge. Right after Node3 (Auditor Agent) finishes, LangGraph calls route_afer_audit(state).

Node 4 HITL Approval Agent:
    (user_input = input(...).strip().lower()):
        input pauses Python script in terminal and waits for approver (Director) to type y or n.
        strip() removes leading and trailing whitespace.
'''

'''
Building & Compiling LangGraph Workflow
Think of StateGraph(SwarmState) as drawing a flowchart on a whiteboard.

workflow.add_node("name", function) draws a Box (Node) on the whiteboard and assigns a Python function to it.
workflow.add_edge("from_box", "to_box") draws an Arrow (Edge) connecting two boxes.

'''


#Shared state between all nodes/swarms in LangGraph workflow
class SwarmState(TypedDict):
    claim: dict
    erp_findings: str
    contract_findings: str
    audit_decision: AuditDecision
    human_approval_status: str
    auto_hitl: bool
    

#Node1: ERP & Accrual Ledger Agent (SQLite Tools only)
def erp_ledger_agent_node(state: SwarmState) -> dict:
    """Queries SQLite TableB (ledger) and TableA (quarterly schedule) for hard ERP numbers."""
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
    erp_agent = create_react_agent(
        model=llm,
        tools=[get_element_ledger, get_quarterly_schedule],
        prompt = (
            "You are the ERP & Accrual Ledger Specialist Agent. "
            "Your ONLY job is to query SQLite using 'get_element_ledger' (and 'get_quarterly_schedule' if the claim involves quarterly case volume or a fiscal quarter). "
            "Report the exact contract_status, remaining_balance, actual_cases, rates and review_notes. "
        ),
    )
    
    user_msg = f"Retrieve all SQLite ERP ledger and quarterly schedule data for this claim:\n{json.dumps(state['claim'], indent=2)}"
    result = erp_agent.invoke({"messages": [("user", user_msg)]})
    
    #Grab the agent's final text message and write it to state[erp_findings]
    erp_report = result["messages"][-1].content
    return {"erp_findings": erp_report}

#Node2: Contract & Policy Agent (ChromaDB RAG tool only)
def contract_policy_agent_node(state: SwarmState) -> dict:
    """Searches ChromaDB for contract clauses and governance rules using claim + ERP context."""
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
    contract_agent = create_react_agent(
        model=llm,
        tools=[search_customer_contract],
        prompt = (
            "You are the CTM Contract & Governance Policy Specialist Agent."
            "Your ONLY job is to call 'search_customer_contract' to retrieve the contract rules"
            "for the customer's element_id (eligible vs prohibited expenes, banner/region exclusions, "
            "tiered_volume thresholds, coupon clearinghouse rules, YOY baselines, and unsigned contract rules)."
            "Summarize the exact contract clauses taht apply to this claim and the ERP numbers"
        ),
    )
    
    user_msg = (
       f"Claim:\n{json.dumps(state['claim'], indent=2)}\n\n"
        f"ERP Findings from Agent 1:\n{state['erp_findings']}\n\n"
        "Search the customer contract in ChromaDB and report all eligibility, rate tier, and governance rules."
    )
    
    result = contract_agent.invoke({"messages": [("user", user_msg)]})
    
    contract_report = result["messages"][-1].content
    return {"contract_findings": contract_report}


#Node3: Financial Auditor & COA Supervisor Agent (Synthesizes A2A Findings -> AuditDecision)
def financial_auditor_node(state: SwarmState) -> dict:
    """Evaluates the CTM Checklist and Chart of Authority using ERP + Contract findings."""
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    #Uses wtih_structured_output() since all tool calls were already done by Agents 1 & 2.
    auditor_llm = llm.with_structured_output(AuditDecision)
    
    auditor_prompt = f"""You are the Senior CTM Financial Auditor & Chart of Authority (COA) Supervisor.
                    Review the incoming claim along with the verified reports from Agent 1 (SQLite ERP Ledger) and Agent 2 (ChromaDB Contract Policy).
                
                Incoming Claim:
                {json.dumps(state['claim'], indent=2)}
                
                Agent 1 Report (SQLite ERP & Accrual Ledger Findings):
                {state['erp_findings']}
                
                Agent 2 Report (ChromaDB Contract & Governance Policy Findings):
                {state['contract_findings']}
                
                Evaluate the 6-Point CTM Checklist and apply the Chart of Authority (COA):
                - Checklist Item #1 (Valid Contract): True if SIGNED; False if UNSIGNED past June 30.
                - Checklist Item #2 (Sufficient Funds): True if remaining_balance >= amount_requested (and not blocked by an upfront advance).
                - Checklist Item #3 (Receipt/POP Attached): True if receipt_or_pop_attached is True.
                - Checklist Item #4 (Expense Eligible): True if the expense item, store banner/region, and package are eligible per Agent 2's contract report.
                - Checklist Item #5 (Math & Volume Verified): True if claimed cases match ERP actual_cases, claimed tier rate is earned, and coupon claims use redeemed (not printed) coupons.
                
                - Checklist Item #6 (Chart of Authority Routing):
                * If ANY of Items #2, #3, #4, or #5 is False -> decision = 'REJECT', required_approver = 'None', verified_payout_amount = 0.0
                * If Items #2-#5 are True, but Item #1 is False (UNSIGNED past June 30) -> decision = 'ESCALATE', required_approver = 'Director of Trade Finance', verified_payout_amount = amount_requested
                * If ALL Items #1-#5 are True:
                    - <= $2,500.00 -> decision = 'APPROVE', required_approver = 'CTM Accountant', verified_payout_amount = amount_requested
                    - $2,500.01 to $50,000.00 -> decision = 'ESCALATE', required_approver = 'CTM Manager', verified_payout_amount = amount_requested
                    - $50,000.01 to $100,000.00 -> decision = 'ESCALATE', required_approver = 'Director of Trade Finance', verified_payout_amount = amount_requested
                    - > $100,000.00 -> decision = 'ESCALATE', required_approver = 'VP of Finance', verified_payout_amount = amount_requested
"""
    decision = auditor_llm.invoke(auditor_prompt)
    
    #Default statue for Approve/Reject (will be overwritten if routed to HITL)
    status = "AUTO_APPROVED" if decision.decision == "APPROVE" else "AUTO_REJECTED"
    return {"audit_decision": decision, "human_approval_status": status}
 
            

#Conditional Edge: Does this claim require HITL esacalation?
def route_after_audit(state: SwarmState) -> str:
    """Routes ESCALATE descisions to the HITL; finishes APPROVE/REJECT immediately."""
    if state["audit_decision"].decision == "ESCALATE":
        return "hitl_agent"
    return END
  
#Node 4: HITL Chart of Authority
def hitl_approval_node(state: SwarmState) -> dict:
    """Pauses workflow to request sign-off from CTM Manager, Director, or VP of Finance."""
    decision = state["audit_decision"]
   
    """If running automated evaluation benchmark, auto-approve w/o waiting for keyboard input"""
    """Ensuring 15-claim benchmark does not freeze waiting for user to type y/n at on every escalate claim."""
    if state.get('auto_hitl'):
        return {'human_approval_status': f'AUTO_ESCALATED_TO ({decision.required_approver})'}
    
    print(f'\n  [HITL TRIGGERED] Escalated to : {decision.required_approver}')
    print(f'  [HITL REASON]  {decision.rationale}')
        
    
    user_input = input(
        f"   Action Required ({decision.required_approver}): Approve ${decision.verified_payout_amount:,.2f} payout? (y/n): "
    ).strip().lower()
    
    
    if user_input =='y':
        return {'human_approval_status' : f'APPROVED_BY ({decision.required_approver})'}
    else:
        decision.verified_payout_amount = 0.0
        return{
            "audit_decision"  : decision,           
            "human_approval_status" : f'REJECTED_BY ({decision.required_approver})'
        }
        

#Assemble and Compile LangGraph StateGraph workflow
def build_swarm_graph():
    workflow = StateGraph(SwarmState)
    
    #Register Nodes
    workflow.add_node('erp_agent', erp_ledger_agent_node)
    workflow.add_node('contract_agent', contract_policy_agent_node)
    workflow.add_node('auditor_agent', financial_auditor_node)
    workflow.add_node('hitl_agent', hitl_approval_node)
    
    #Connect Nodes
    workflow.add_edge(START, 'erp_agent')
    workflow.add_edge('erp_agent', 'contract_agent')
    workflow.add_edge('contract_agent', 'auditor_agent')

    #Connect conditional edge after Agent 3 (either HITL or END) 
    workflow.add_conditional_edges('auditor_agent', route_after_audit)
    workflow.add_edge('hitl_agent', END)
    
    
    return workflow.compile()

def main() -> None:
    with open(CLAIMS_PATH, "r", encoding="utf-8") as f:
        claims = json.load(f)
    swarm = build_swarm_graph()
    # Test 3 diverse claims:
    # - claims[0] (CLM-HESS-001: $1,850 JBP Dinner -> Auto-Approve <= $2,500)
    # - claims[6] (CLM-BP-002: $6,400 Midwest BP Connect -> Auto-Reject Banner Exclusion)
    # - claims[7] (CLM-WAWA-001: $1,950 Unsigned Contract -> Triggers HITL Director Escalation!)
    test_claims = [claims[0], claims[6], claims[7]]
    for claim in test_claims:
        clean_claim = {
            k: v for k, v in claim.items() if k not in ("expected_outcome", "expected_approver")
        }
        
        print(f"\n{'='*70}")
        print(f"SWARM AUDITING: {claim['claim_id']} ({claim['customer_name']} - {claim['element_id']})")
        print(f"Requested: ${claim['amount_requested']:,.2f} | Expected: {claim['expected_outcome']} ({claim['expected_approver']})")
        
        final_state = swarm.invoke({"claim": clean_claim})
        decision = final_state["audit_decision"]
        
        print(f"\n-> 1. ERP Agent Report:      {final_state['erp_findings'][:140]}...") #print 1st 140 chars of Agent 1 and 2 reports.
        print(f"-> 2. Contract Agent Report: {final_state['contract_findings'][:140]}...")
        print(f"-> 3. Auditor Decision:      {decision.decision} (Required Approver: {decision.required_approver})")
        print(f"-> 4. Final Workflow Status: {final_state['human_approval_status']}")
        print(f"-> 5. Verified Payout:       ${decision.verified_payout_amount:,.2f}")
        print(f"-> 6. Checklist:             [1.Contract:{decision.item_1_valid_contract} | 2.Funds:{decision.item_2_sufficient_funds} | 3.Receipt:{decision.item_3_receipt_or_pop_attached} | 4.Eligible:{decision.item_4_expense_eligible} | 5.Math/Vol:{decision.item_5_math_and_volume_verified}]")
        print(f"-> 7. Rationale:             {decision.rationale}")


if __name__ == "__main__":
    main()