import json
import sqlite3
from pathlib import Path
from langchain_core.tools import tool

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

#DB_PATH = Path("data/ctm_ledger.db")

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

'''
search_customer_contract() queries ChromaDB for customer CTM contracts using a query string.

embeddings = OpenAIEmbeddings(model="text-embedding-3-small"):
    Initializes same embedding model used in ingest_contracts.py.
    Best to always search a vector db using same embedding model used to index it. This helps query
    number vector lives in same mathematical space as the stored chunks. 

vector_store = Chroma(): 
    Calling Chroma(...) instad of Chroma.from_documents(...).
    Chroma.from_documents() writes new chunks to disk. (used in ingest_contracts.py)
    Chroma() just opens the existing db sitting in data/chroma_db under collection_name = "ctm_contracts_and_policy".

results = vector_store.similarity_search(query, k=3):
    Using CLM-BP-002 (a Midwest BP Connect foodservice combo claim) as an example...
    Agent calls search_customer_contract(query="BP-COMBO eligible banners Midwest BP Connect ampm").
    
    4 thinks occuring under hood:
        Embed the query: Chroma send string "BP-COMBO eligible banners Midwest BP Connect ampm" to OpenAI
        text-embedding model, which turns the seach question into a vecotr of numbers.
        
        Compare against all chunks: Chroma compare the query vector agains all contract/policy vectors 
        stored in data/chroma_db and calculates a mathematical similarity score (cosine distance) for each
        chunk - measuring how close the 'meaning' of each chunk is to the query. 
        
        Rank & Pick the Top: k=3 means Top K nearest neighbors:
        Instead of just returning 1 chunk (which might miss a general contract header or governace rule) or 
        ALL chunks (which wastes tokesn and clutters the prompt), k=3 grabs the 3 most relevant chunks ranked 
        from best match (#1) to 3rd-best match (#3).
        
        Retrun a list of 3 Document objects (results):
        Results is a list containing 3 LangChain Document objects [doc1, doc2, doc3].
        Each do object has 2 attributes:    
            doc.page_content: The actual Markdown text of that contract section. 
            doc.metadata: A dictionary of tags attached during ingestion (source_file, section, subsection).
    
    The for loop is needed b/c an LLM tool cannot return raw Python Document objects. LLM needs a single 
    clean string containing the text and source filename of all 3 chunks. 
    
    Create a empty list that will hold 3 formatted text strings, one for each of the 3 retrieved chunks. 
    for doc in results:
        loops thru the 3 Document objects returned by similiarty_search(query, k=3), one doc at a time.
        Pass1 = bestmatch, Pass2 = 2nd bestmatch, Pass3 = 3rd bestmatch.
        
     source = doc.metadata.get("source_file", "unknown"):
        looks inside doc.metadata for source_file key stamped in ingest.contracts.py (e.g. "hess_contract_2026.md").
        get('source_file', 'unknown') is defensive coding in case source_file is not in doc.metadata. 
        Will not crash with a Keyerror, will just return "unknown".

    section = doc.metadata.get("subsection") or doc.metadata.get("section", ""):
        Tries to grab the most specific header first like subsection, 
        which is the ### Element title like "3. Element 203 — ampm West Coast Foodservice Combo". 
        If a chunk came from an intro section that only had a ## header (so "subsection" doesn't exist and returns None), 
        or operator automatically falls back to doc.metadata.get("section", "").
    
    
    formatted_chunks.append(f"[{source} | {section}]\n{doc.page_content}"):
        Builds a single string with a clear citation header on top ([bp_contract_2026.md | 3. Element 203...]), 
        a newline \n, and the full contract text (doc.page_content), and adds it to the formatted_chunks list.
    
    return "\n\n---\n\n".join(formatted_chunks):
        join() glues the 3 strings in formatted_chunks together into a single string, placing a visual divier (---)
        between Chunk1, Chunk2 and Chunk3 so gpt-4o-mini can easly tell where on contract chunks ends and the next begins. 
        
        [bp_contract_2026.md | 3. Element 203 — ampm West Coast Foodservice Combo]
        ### 3. Element 203 — ampm West Coast Foodservice Combo
        - **Element ID:** `BP-COMBO`
        - **Eligible Banners:** Strictly West Coast *ampm* stores...
        ---
        [bp_contract_2026.md | Section 1: Company History & Account Background]
       ...

'''


load_dotenv()  

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data/ctm_ledger.db"
CHROMA_DIR = ROOT_DIR / "data/chroma_db"


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


@tool
def search_customer_contract(query: str) -> str:
    """
    Search customer CTM contracts and the CTM GovernancePolicy in ChromaDB.
    Use this tool to check Checklist Item #4 (eligible vs. ineligible expenses, store banner exclusions,
    coupon clearinghourse rules, tiered volume thresholds, YOY growth baselines and Chart of Authority rules.)   
    """
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="ctm_contracts_and_policy",
    )

    results = vector_store.similarity_search(query, k=3)

    formatted_chunks = []
    for doc in results:
        source = doc.metadata.get("source_file", "unknown")
        section = doc.metadata.get("subsection") or doc.metadata.get("section", "")
        formatted_chunks.append(f"[{source} | {section}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_chunks)