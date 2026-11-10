import sqlite3
from pathlib import Path
import json

'''
Both TableA and TableB are populated by data/seed_data.json. The seed data is a mockup of CTM customer program. 
The seed_data.json file contains 2 list of dictionaries:
    quarterly_schedule: reflects qtrly accrual reviews. Each qtr updated with actual volume, revised rates, revised accrual dollars. 
    elements_ledger: contains element, supporting packages, contract status, q1-q4 rates, ytd actual cases, ytd_accrued_budget, 
                     ytd_spend, remaining_balance.
                     
'''
#Path to SQLite database file and seed JSON (contains mock quarterly schedule rows(Table A) and element ledger rows (TableB))
#DB_PATH = Path('data/ctm_ledger.db')
#SEED_DATA_PATH = Path('data/seed_data.json')

#Climb up 2 levels from __file__. Then glues/joins the rest of the path to get to the data folder and its subfolders. 
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data/ctm_ledger.db"
SEED_FILE = ROOT_DIR / "data/seed_data.json"

def create_tables(conn: sqlite3.Connection) -> None:
    """ Create Table A (quarterly schedule) and Table B (YTD elements ledger)"""
    cursor = conn.cursor()
    
    #Drop existing tables if if rerun seed script
    cursor.execute("DROP TABLE IF EXISTS element_quarterly_schedule")
    cursor.execute("DROP TABLE IF EXISTS accrual_elements_ledger")
    
    #Table A; quarterly volume, accrual rates, bottler review notes
    cursor.execute("""
            CREATE TABLE element_quarterly_schedule (
            schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            customer_name TEXT NOT NULL,
            element_id TEXT NOT NULL,
            quarter TEXT NOT NULL,
            supporting_packages TEXT NOT NULL,
            planned_cases INTEGER NOT NULL,
            actual_cases INTEGER NOT NULL,
            contract_base_rate REAL NOT NULL,
            quarterly_accrual_rate REAL NOT NULL,
            quarterly_accrued_dollars REAL NOT NULL,
            review_notes TEXT
        )
        """)
    
    #Table B: YTD accrual ledger (Dashbaord & Balance verification table) 
    cursor.execute("""
          CREATE TABLE accrual_elements_ledger (
            element_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            customer_name TEXT NOT NULL,
            contract_status TEXT NOT NULL,
            element_name TEXT NOT NULL,
            supporting_packages TEXT NOT NULL,
            base_rate REAL NOT NULL,
            q1_rate REAL, 
            q2_rate REAL,
            q3_rate REAL,
            q4_proj_rate REAL,
            ytd_actual_cases INTEGER NOT NULL,
            ytd_accrued_budget REAL NOT NULL,
            ytd_spend REAL NOT NULL,
            remaining_balance REAL NOT NULL
        )         
        """)
    conn.commit()
    print("Tables created successfully.")

def seed_data(conn: sqlite3.Connection) -> None:
    """Load rows form seed_data.json and insert into Table A and Table B"""  
    with open(SEED_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)    
    cursor = conn.cursor()
    
    # Insert into Table A: element_quarterly_schedule
    cursor.executemany("""
        INSERT INTO element_quarterly_schedule (
            customer_id, customer_name, element_id, quarter,
            supporting_packages, planned_cases, actual_cases,
            contract_base_rate, quarterly_accrual_rate,
            quarterly_accrued_dollars, review_notes
        ) VALUES (
            :customer_id, :customer_name, :element_id, :quarter,
            :supporting_packages, :planned_cases, :actual_cases,
            :contract_base_rate, :quarterly_accrual_rate,
            :quarterly_accrued_dollars, :review_notes
        )
    """, data["quarterly_schedule"])
    
    # Insert into Table B: accrual_elements_ledger
    cursor.executemany("""
        INSERT INTO accrual_elements_ledger (
            element_id, customer_id, customer_name, contract_status,
            element_name, supporting_packages, base_rate,
            q1_rate, q2_rate, q3_rate, q4_proj_rate,
            ytd_actual_cases, ytd_accrued_budget, ytd_spend, remaining_balance
        ) VALUES (
            :element_id, :customer_id, :customer_name, :contract_status,
            :element_name, :supporting_packages, :base_rate,
            :q1_rate, :q2_rate, :q3_rate, :q4_proj_rate,
            :ytd_actual_cases, :ytd_accrued_budget, :ytd_spend, :remaining_balance
        )
    """, data["elements_ledger"])
    conn.commit()
    print(f"Inserted {len(data['quarterly_schedule'])} quarterly rows into Table A.")
    print(f"Inserted {len(data['elements_ledger'])} element ledger rows into Table B.")
    
def main() -> None:
    #Ensure data directory exists. Create if it doesn't. 
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with sqlite3.connect(DB_PATH) as conn:
        create_tables(conn)
        seed_data(conn)
        
        
if __name__ == "__main__":
    main()