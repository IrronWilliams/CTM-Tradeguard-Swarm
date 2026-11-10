from typing import Literal
from pydantic import BaseModel, Field

''' 
When CTM Auditor Agent finishes checking a claim, I dont want the agent to ramble in a 
free from approach/paragraph. I want the agent to complete teh CTM Checklist and COA as a
strict, predicatable JSON object that can be read by a dashboard and/or evaluation scripts. 

BaseModel from Pydantic turns a Python class into a strict data validator. When pass response to 
LangChain Agent, got-40-mini is forced to return these exact fields. 

Literal["Approve", "Reject", "Escalate"] locks the field to only 3 possible values. The LLM 
is physically provented from inventing an additonal status like 'Pending" or 'Maybe"

Just like the doctstring in @tool, Pydantic sends each Field(description=...) to the LLM so it knows
the exact business rule for filling out that field. 

Creating fields for AuditDecision, which powers 3 places in project. 
In current phase (terminal), agent.py script will print each field of AuditDecison in terrminal when 
it finishes an audit claim.

In Evaluation Pipeline phase, automated evaluation script will compare decision ("Approve", "Reject", "Escalate")
and required_approver directly agains the ground-truth expected outcome in sample_claims.json to score 
the agent's accuracy.

In Dashboard phase, dashboard will take some of these fields and display them in the UI.

The LLM fills out the AuditDecision fields based on the CTM Payment Checklist and Chart of Authority (COA) rules.
'''
class AuditDecision(BaseModel):
    """
    Structured CTM Payment Checklist and Chart of Authority audit report.
    """
    claim_id: str= Field(description="The ID of the claim being audited (e.g. CLM-HESS-001)")
    customer_id: str=Field(description="The customer ID (eg. CUST-HESS-001)")
    element_id: str=Field(description="The element ID (e.g. HESS-PROMO)")
    
    #CTM Payment Checklist (True = Pass, False = Fail/Exception)
    item_1_valid_contract: bool=Field(
        description="True if valid, SIGNED contract is on file. Fasle if contract is UNSIGNED pasat June 30." 
    )
    
    item_2_sufficient_funds: bool=Field(
        description="True if element is active and remaining balance >= amount requested and not blocked by an upfront advance)."
    )
    
    item_3_receipt_or_pop_attached: bool=Field(
        description="True if required itemized receipt, audit report, or clearninghouse summary attached."
    )
    
    item_4_expense_eligible: bool=Field(
        description="True if expense type, banner/region and package are eligible under element terms."
    )
    
    item_5_math_and_volume_verified: bool=Field(
        description="True if claimed math, tier rate, redeemed coupon count and ERP actual cases match"
    )
    
    #Chart of Authority & Final Decisioin
    verified_payout_amount: float =Field(
        description="The verified dollar amount eligible for payout (0.0 if rejected)."
    )
    
    decision: Literal["APPROVE", "REJECT", "ESCALATE"] = Field(
        description="APPROVE (<= $2,500 & signed contract), ESCALATE (> $2,500 or unsigned contract past June 30), or REJECT (if any checklist item 2-5 fails)."
    )
    required_approver: Literal[
        "CTM Accountant",
        "CTM Manager",
        "Director of Trade Finance",
        "VP of Finance",
        "None",
    ] = Field(
        description="Required approver per Chart of Authority (or 'None' if REJECT)."
    )
    rationale: str = Field(
        description="Clear accounting explanation citing the specific CTM Checklist items, ERP numbers, and COA rules."
    )