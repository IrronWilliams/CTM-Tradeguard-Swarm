import json
import sqlite3
from pathlib import Path
from langchain_core.tools import tool

#DB_PATH = Path("data/ctm_ledger.db")
#Climb up 2 levels from __file__. Then glues/joins the rest of the path to get to the data folder and its subfolders. 
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data/ctm_ledger.db"

'''  
@tool from langchain turns a normal Python function into an AI tool. With the required docstring, 
LangChain sends the docstring and the type hints to the LLM (gpt-4o-mini) so the LLM knows when and 
how to call the tool. 

Sturcturally the 2 functions follow the same 4 step SQLite pattern:
    (Connect->Configure Row Factory-> Run SELECT Query-> Convert to JSON).
    
get_element_ledger() queries Table B (accrual_element_ledger) using 1 filter (element_id).
The agent uses this on every claim to check checklist Item #1 (contract_status is signined/unsigned) and
checklist Item #2 (remaining_balance is enough money in the element to cover the expense)

get_quarterly_schedule() queries Table A (element_quarterly_schedule) using 2 filters (element_id and quarter).
The agent uses this when a retailer submits a quarterly volume rebate to check checklist Item #5
(actual cases shipped in that specific quarter and teh bottler review notes). 


@tool tells LangChain to wrap function into a schema OpenAI gpt-4o-mini can call. 
The typehint (element_id: str) tells the LLM that the input paramter element)id must be a string. 
->str is a type hint for a string object. Tells python and fellow devs that function returns a string, 
specifically a JSON-formatted string the LLM can read. 

@tool needs the docstring, it sends this text directly to the LLM so model knows what data lives in the
tool.

Opens a connection to the SQLite database file using a 'with' context manager so the connection closes
automatically when done. By default, SQLlite retuns rows as plain tuples w/o column names. 
Setting conn.row_factory=sqlite3.Row attaches the SQL column names to each value so 
I can convert the row into a Python dictionary, dict(row). 

The cursor creates a cursor object that executes SQL queries. 
Where element_id=? and (element_id,): in Python, a 1-item tuple must have a trailing comma, otherwise
Python treats element_id as just a striing in parentheses.
row=cursor.fetchone(): Since element_id is the PK of TableB, there can only be 1 matching row. 
fetchone() grabs that single row (or returns None if no match exists).

if not row: handlse missing elements. If the agent passes an invalid element_id, (row is None), instead
of crashing program, return a clean JSON error message to the LLM.

function returns the result as a formatted JSON. dict(row) converts the SQLite row into a Python
dictionary and json.dumps() turns the dictionary into a formatted JSON string for the LLM.
indent=2 tells Pyton to 'pretty print' the JSON string across multiple lines with 2 spaces of 
indentation, instead of squishing the entire dict onto a single long hard to read line. 
indent=2 makes the tool output much easier to read in the terminal and in LangSmith traces. 

W/o indent=2 (json.dumps(dict(row)))
{"element_id": "HESS-PROMO", "customer_id": "CUST-HESS-001", "contract_status": "SIGNED", "remaining_balance": 40165.0}

With indent=2 (json.dumps(dict(row), indent=2))
{
  "element_id": "HESS-PROMO",
  "customer_id": "CUST-HESS-001",
  "contract_status": "SIGNED",
  "remaining_balance": 40165.0
}

'''


@tool
def get_element_ledger(element_id: str) -> str:
    """ 
    Query Table B (accrual_elements_ledger) for a specific element_id.
    Returns contract_status, supporting_packages, rates, ytd_actual_cases, 
    ytd_accrued_budget, ytd_spend and remaining_balance.
    
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor=conn.cursor()
        cursor.execute(
            "SELECT * FROM accrual_elements_ledger WHERE element_id = ?", (element_id,), 
        )
        row = cursor.fetchone()
        
    if not row:
        return json.dumps({"error": f"Element '{element_id} not found in ledger"})
    
    return json.dumps(dict(row), indent=2)

@tool
def get_quarterly_schedule(element_id: str, quarter: str) -> str:
    """
    Query Table A (element_quarterly_schedule) for a specific element_id and quarter (e.g 2026-Q3).
    Returns planned_cases, actual_cases, contract_base_rate, quarterly_accrual_rate, 
    quarterly_accrued_dollars and review_notes.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor=conn.cursor()
        cursor.execute(
            "SELECT * FROM element_quarterly_schedule WHERE element_id = ? AND quarter = ?", 
            (element_id, quarter), 
        )
        row = cursor.fetchone()
    
    if not row:
        return json.dumps(
            {"error": f"No schedule found for element '{element_id} in quarter '{quarter}'"}
        )
    
    return json.dumps(dict(row), indent=2)


