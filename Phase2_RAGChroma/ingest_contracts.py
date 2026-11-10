import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter


'''
import shutil: shell utilities module for high level file/folder operations. Will use to delete a folder. 
Chroma: vector db class that stores and searches text chunks. 
OpenAIEmbeddings: calls text embedding model to convert text paragraphs into numberical vectors (list of numbers) that represent text 'meaning'.
MarkdownHeaderTextSplitter: a splitter that knows Markdown structure (#, ##, ###). Cuts documents cleanly between sections instead of mid sentence.
                            also determines the exact number of chunks. 

load_and_split_markdown_files: 
    headers_to_split_on []: tells the splitter to start a new chunk whenever it sees # (Main Title),
                            ## (Section), or ### (Individual element like ### Element 202 - 12Pack Case Pack). 
                            Also saves the title into each chunk's metadata dictionary. 

splitter: strip_headers=True means the splitter deletes the #, ## and ### header lines from the actual chunk text (page_content).
          strip_headers=False keeps the header lines in the chunk text. So element name (like ### 3. Element 203 — ampm West Coast Foodservice Combo) 
          stays right inside the text that gets embedded and accessible for the agent. 
          
          
Function loops thru the the markdown files. Think of the for loops taking printed contracts and cutting each contract into individual sections,
stamps the contracts filename on tope of each cut piece and saves into one master folder. The loop manages this by:
    a loop inside a loop (an outer loop for each file, and an inner loop for each cut chunk of that file).

(CONTRACTS_DIR.glob('*.md')): Finds all .md files inside data/contracts/ and combines them (+) 
with [GOVERNANCE_FILE] into a single list of file paths.

Example of iteration of loop --> Outer Loop (file_path = hess_contract_2026.md):
    Python reads hess_contract_2026.md into text.
    splitter.split_text(text) cuts the Hess contract by its #, ##, ### headers into say 6 pieces (chunks is a list of 6 Hess chunks).

    The Inner Loop (for chunk in chunks:) runs 6 times—stamping "source_file": "hess_contract_2026.md" onto each of those 6 Hess chunks.
    all_chunks.extend(chunks) drops all Hess chunks into all_chunks. (now all_chunks has 6 items).
    Prints: Loaded hess_contract_2026.md -> 6 chunks.

Pass #2 of the Outer Loop (file_path = bp_contract_2026.md):
    Loop iterates/moves to the second file: bp_contract_2026.md.
    It cuts the BP contract into 6 new pieces (so the variable chunks now holds the 6 BP chunks).
    The Inner Loop (for chunk in chunks:) stamps "source_file": "bp_contract_2026.md" onto each of the 6 BP chunks.
    all_chunks.extend(chunks) adds the 6 BP chunks into all_chunks alongside the Hess chunks (now all_chunks has 6 + 6 = 12 items).
    Prints: Loaded bp_contract_2026.md -> 6 chunks.

MarkdwonHeader Text Splitter: determines the nubmer of chunks by counting the #, ##, and ### secction headers, specifically how many elements
                              were inside each contract. 

Example of how many chunks are created from a contract.
Every time MarkdownHeaderTextSplitter saw a #, ##, or ### heading in hess_contract_2026.md, it made a cut:
    Chunk 1 (#): Top Contract Header (Customer ID CUST-HESS-001, Contract Status SIGNED, Participating Locations)
    Chunk 2 (## Section 1): Company History & Account Background
    Chunk 3 (### 1. Element 101): HESS-PROMO (General Promotions rules)
    Chunk 4 (### 2. Element 102): HESS-20OZ (Cold Vault Rebate rules)
    Chunk 5 (### 3. Element 103): HESS-SUMMER (Summer Endcap rules)
    Chunk 6 (### 4. Element 104): HESS-COUPON (Fuel-Pump Coupon & Clearinghouse rules)
    Total: 2 intro sections + 4 Elements = 6 chunks.   
    
    CTM Governance Policy (6 chunks): Cut by its 6 # / ## / ### policy sections = 6 chunks
    
Because each element was cut into its own chunk, when the agent searches ChromaDB for 'BP=Combo' or 'CK=200z-Inc', 
ChromaDB hands the agent the exact elements complete rule block w/o mixing in unrelated elements from the same contract. 
This is important because the agent needs to know the exact rules for each element, not a mix of rules from other elements.    
                              
main(): builds & persists the vector database.
If data/chrom_db exists from a previous run, shutil.rmtree() deletes folder, to prevent storing duplicate copies of contracts. 
calls load_and_split_markdown_files() to get all the chunks from the markdown files.
Chroma.from_documents(): sends each chunk to OpenAI's text-embedding model to calculate its vector embedding, and saves the text, 
                        metadata, and vectors onto hard drive inside data/chroma_db folder, under name ctm_contracts_and_policy.
                    
                        

'''



#load environment variables from .env file
load_dotenv()

#CHROMA_DIR = Path('data/chroma_db')
#CONTRACTS_DIR = Path('data/contracts')
#GOVERNANCE_FILE = Path('data/governance/ctm_governance_policy.md')

#Climb up 2 levels from __file__. Then glues/joins the rest of the path to get to the data folder and its subfolders. 
ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / 'data/chroma_db'
CONTRACTS_DIR = ROOT_DIR / 'data/contracts'
GOVERNANCE_FILE = ROOT_DIR / 'data/governance/ctm_governance_policy.md'

def load_and_split_markdown_files() -> list:
    """Read all contract and governance markdown files and split by section headers"""
    headers_to_split_on = [
        ('#', 'document_title'),
        ('##', 'section'),
        ('###', 'subsection'),
    ]
    
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    
    #Collect all customer contract + governance policy file
    md_files = list(CONTRACTS_DIR.glob('*.md')) + [GOVERNANCE_FILE] #list all files 
    all_chunks = []  #empty master list to hold all chunks from all files
    
    """Outer loop picks one md file at a time. Reads the whole file (text) and splits into chunks (chunks)"""
    """Inner loop stamps the filename ont each chunk from current iteration of outer loop."""   
    for file_path in md_files:
        text=file_path.read_text(encoding='utf-8')
        chunks=splitter.split_text(text)

        #Inner Loop
        for chunk in chunks:
            chunk.metadata['source_file'] = file_path.name
            
        #Still inside outer loop, after inner loop finishes, add all chunks from current file to master list
        all_chunks.extend(chunks)
        print(f"Loaded {file_path.name} -> {len(chunks)} chunks")
    
    return all_chunks #return full list after all files are finhsed


def main() ->None:
    #Delete existing chroma_db folder with each script run. This prevents duplicate chunks.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        
    chunks = load_and_split_markdown_files()
    
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name='ctm_contracts_and_policy',
    )
    
    print(f'\nSuccessfully indexed {len(chunks)} total chunks into {CHROMA_DIR}.')
    

if __name__ == "__main__":
    main()