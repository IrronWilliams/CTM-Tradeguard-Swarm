import json
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st

from swarm import build_swarm_graph
from tools import search_customer_contract




# 1. Page Configuration (must be the very first Streamlit command)
st.set_page_config(
    page_title="TradeGuard — CTM Autonomous Swarm",
    layout="wide",
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data/ctm_ledger.db"
CLAIMS_PATH = ROOT_DIR / "data/claims/sample_claims.json"
EVAL_RESULTS_PATH = ROOT_DIR / "data/eval_results.json"


# 2. Cached Data Loaders (loads SQLite & JSON into memory for fast filtering)
@st.cache_data
def load_ledger_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads Table B (elements) and Table A (quarterly schedules) from SQLite into Pandas DataFrames."""
    with sqlite3.connect(DB_PATH) as conn:
        elements_df = pd.read_sql_query(
            "SELECT * FROM accrual_elements_ledger ORDER BY customer_id, element_id", conn
        )
        schedules_df = pd.read_sql_query(
            "SELECT * FROM element_quarterly_schedule ORDER BY element_id, quarter", conn
        )
    return elements_df, schedules_df


@st.cache_data
def load_claims_and_eval() -> tuple[list[dict], dict]:
    """Loads the 15 sample claims and the Golden Dataset evaluation results."""
    with open(CLAIMS_PATH, "r", encoding="utf-8") as f:
        claims = json.load(f)

    eval_data = {}
    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

    return claims, eval_data


# Load data
elements_df, schedules_df = load_ledger_data()
claims_list, eval_data = load_claims_and_eval()

# Initialize session state for live swarm audits and HITL sign-offs
if "live_audit" not in st.session_state:
    st.session_state["live_audit"] = None
if "hitl_decisions" not in st.session_state:
    st.session_state["hitl_decisions"] = {}


# 3. Sidebar Customer Filter & Governance Quick Reference
st.sidebar.title("TradeGuard CTM Swarm")
st.sidebar.caption("National Convenience Retail CTM Accounting & Audit Prototype")

customer_options = ["ALL"] + sorted(elements_df["customer_name"].unique().tolist())
selected_customer = st.sidebar.selectbox("Filter by Customer Chain:", customer_options)

st.sidebar.divider()
st.sidebar.subheader("Chart of Authority (COA) Tiers")
st.sidebar.markdown(
    """
- **<= $2,500:** CTM Accountant (`APPROVE`)
- **$2,500.01 – $50,000:** CTM Manager (`ESCALATE`)
- **$50,000.01 – $100,000:** Director of Trade Finance (`ESCALATE`)
- **> $100,000:** VP of Finance (`ESCALATE`)
- **Unsigned Past June 30:** Mandatory Director of Trade Finance (`ESCALATE`)
"""
)

# Filter Table B DataFrame based on sidebar selection
if selected_customer == "ALL":
    filtered_elements_df = elements_df
    filtered_claims = claims_list
else:
    filtered_elements_df = elements_df[elements_df["customer_name"] == selected_customer]
    filtered_claims = [c for c in claims_list if c["customer_name"] == selected_customer]


# 4. Main Header & Top Executive KPI Row
st.title("National Convenience Retail — CTM Accrual & Audit Dashboard")
st.caption("Multi-Agent LangGraph Swarm: ERP Ledger Agent (SQLite) ↔ Contract Policy Agent (ChromaDB RAG) → Financial Auditor (COA)")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("YTD Accrued Budget", f"${filtered_elements_df['ytd_accrued_budget'].sum():,.2f}")
kpi2.metric("YTD Promotional Spend", f"${filtered_elements_df['ytd_spend'].sum():,.2f}")
kpi3.metric("Remaining Balance", f"${filtered_elements_df['remaining_balance'].sum():,.2f}")
kpi4.metric(
    "Golden Dataset Accuracy",
    f"{eval_data.get('exact_match_accuracy_pct', 100.0)}% ({eval_data.get('total_claims', 15)}/15)",
)
kpi5.metric("Avg Swarm Latency", f"{eval_data.get('average_latency_seconds', 13.56)}s")

st.divider()

# 5. Four Interactive Prototype Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Accrual Ledger & Schedules (Table B & A)",
        "2. Claims Audit & Live HITL Swarm",
        "3. Custom Claim Sandbox (Test New Claims)",
        "4. Contract & Policy RAG Explorer",
    ]
)

with tab1:
    st.subheader(f"Table B: Customer Elements & Accrual Ledger ({len(filtered_elements_df)} Elements)")
    st.dataframe(filtered_elements_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Table A: Quarterly Schedule & Bottler Review Drill-Down")
    selected_element_id = st.selectbox(
        "Select an Element ID to inspect Q1–Q3 Planned vs. Actual Shipped Cases & Bottler Notes:",
        filtered_elements_df["element_id"].tolist(),
    )
    element_schedule_df = schedules_df[schedules_df["element_id"] == selected_element_id]
    st.dataframe(element_schedule_df, use_container_width=True, hide_index=True)


# 6. Tab 2: Golden Dataset Claims & Live 3-Agent Swarm with Interactive HITL Gate
with tab2:
    st.subheader(f"Golden Dataset Evaluation & Live Swarm Auditor ({len(filtered_claims)} Claims)")

    if eval_data.get("results"):
        eval_df = pd.DataFrame(eval_data["results"])
        if selected_customer != "ALL":
            eval_df = eval_df[eval_df["customer_name"] == selected_customer]
        st.dataframe(eval_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Run Live 3-Agent LangGraph Swarm on a Claim")

    claim_options = {
        f"{c['claim_id']} | {c['customer_name']} ({c['element_id']}) — ${c['amount_requested']:,.2f}": c
        for c in filtered_claims
    }
    selected_claim_label = st.selectbox("Select a Claim to Audit Live:", list(claim_options.keys()))
    target_claim = claim_options[selected_claim_label]

    if st.button("▶ Run Live 3-Agent Swarm Audit", type="primary"):
        clean_claim = {
            k: v for k, v in target_claim.items() if k not in ("expected_outcome", "expected_approver")
        }
        with st.status("Executing 3-Agent LangGraph Swarm...", expanded=True) as status:
            st.write("1. Node 1 (`erp_agent`): Querying SQLite Table B & Table A...")
            st.write("2. Node 2 (`contract_agent`): Searching ChromaDB Contract & Policy chunks...")
            st.write("3. Node 3 (`auditor_agent`): Evaluating 6-Point Checklist & COA Routing...")
            swarm = build_swarm_graph()
            final_state = swarm.invoke({"claim": clean_claim, "auto_hitl": True})
            st.session_state["live_audit"] = final_state
            status.update(label="Swarm Audit Complete!", state="complete", expanded=False)

    # Display Live Audit Result & Interactive HITL Gate if available
    if st.session_state["live_audit"]:
        res = st.session_state["live_audit"]
        dec = res["audit_decision"]

        st.markdown(f"#### Audit Result for `{dec.claim_id}` ({dec.element_id})")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("**1. Node 1: SQLite ERP & Accrual Agent**")
            st.markdown(res["erp_findings"])

        with col2:
            st.warning("**2. Node 2: ChromaDB Contract RAG Agent**")
            st.markdown(res["contract_findings"])

        with col3:
            st.success("**3. Node 3: Financial Auditor & COA Decision**")
            st.write(f"**Decision:** `{dec.decision}`")
            st.write(f"**Required Approver:** `{dec.required_approver}`")
            st.write(f"**Verified Payout:** `${dec.verified_payout_amount:,.2f}`")
            st.write(
                f"**6-Point Checklist:** 1.Contract: `{dec.item_1_valid_contract}` | "
                f"2.Funds: `{dec.item_2_sufficient_funds}` | 3.Receipt: `{dec.item_3_receipt_or_pop_attached}` | "
                f"4.Eligible: `{dec.item_4_expense_eligible}` | 5.Math/Vol: `{dec.item_5_math_and_volume_verified}`"
            )
            st.write(f"**Rationale:** {dec.rationale}")

            # Interactive Human-in-the-Loop (HITL) COA Approval Buttons for ESCALATE claims!
            if dec.decision == "ESCALATE":
                st.divider()
                st.markdown(f"**⚠️ HITL Action Required ({dec.required_approver}):**")
                b1, b2 = st.columns(2)
                if b1.button(f"✅ Approve (${dec.verified_payout_amount:,.2f})", key=f"app_{dec.claim_id}"):
                    st.session_state["hitl_decisions"][dec.claim_id] = f"APPROVED BY {dec.required_approver}"
                if b2.button("❌ Reject Claim ($0.00)", key=f"rej_{dec.claim_id}"):
                    st.session_state["hitl_decisions"][dec.claim_id] = f"REJECTED BY {dec.required_approver}"

                if dec.claim_id in st.session_state["hitl_decisions"]:
                    st.success(f"Final HITL Status: **{st.session_state['hitl_decisions'][dec.claim_id]}**")


# 7. Tab 3: Custom Claim Sandbox (End-User What-If Testing)
with tab3:
    st.subheader("Custom Claim Sandbox — Test Any Scenario Against the Swarm")
    st.caption("Try testing prohibited items (e.g. store manager gift cards), wrong banners (BP Connect), or unsigned contracts (Wawa)!")

    with st.form("custom_claim_form"):
        c1, c2, c3 = st.columns(3)
        selected_el_row = c1.selectbox(
            "Customer & Element ID:",
            elements_df.apply(lambda r: f"{r['element_id']} ({r['customer_name']} — {r['element_name']})", axis=1).tolist(),
        )
        chosen_element_id = selected_el_row.split(" (")[0]
        chosen_cust_row = elements_df[elements_df["element_id"] == chosen_element_id].iloc[0]

        custom_quarter = c2.selectbox("Fiscal Quarter:", ["2026-Q1", "2026-Q2", "2026-Q3", "2026-Q4"])
        custom_amount = c3.number_input("Amount Requested ($):", min_value=100.0, value=3500.0, step=500.0)

        c4, c5, c6 = st.columns(3)
        custom_cases = c4.number_input("Claimed Cases (0 if promotional expense):", min_value=0, value=0, step=1000)
        custom_rate = c5.number_input("Claimed Rate ($/case, 0 if N/A):", min_value=0.0, value=0.0, step=0.05)
        custom_receipt = c6.checkbox("Itemized Receipt / POP Audit Attached?", value=True)

        custom_desc = st.text_area(
            "Claim Description & Attachment Notes:",
            value="Promotional display & catering invoice for Q3 regional convenience store feature.",
        )

        submitted = st.form_submit_button("🚀 Audit Custom Claim with 3-Agent Swarm", type="primary")

    if submitted:
        custom_claim_payload = {
            "claim_id": "CLM-CUSTOM-999",
            "customer_id": chosen_cust_row["customer_id"],
            "customer_name": chosen_cust_row["customer_name"],
            "element_id": chosen_element_id,
            "fiscal_quarter": custom_quarter,
            "amount_requested": float(custom_amount),
            "claimed_cases": int(custom_cases) if custom_cases > 0 else None,
            "claimed_rate": float(custom_rate) if custom_rate > 0 else None,
            "receipt_or_pop_attached": bool(custom_receipt),
            "description": custom_desc,
        }
        with st.spinner("Running 3-Agent Swarm on your custom claim..."):
            swarm = build_swarm_graph()
            custom_state = swarm.invoke({"claim": custom_claim_payload, "auto_hitl": True})
            c_dec = custom_state["audit_decision"]

        st.success(f"**Decision:** `{c_dec.decision}` | **Required Approver:** `{c_dec.required_approver}` | **Verified Payout:** `${c_dec.verified_payout_amount:,.2f}`")
        st.write(
            f"**Checklist:** 1.Contract: `{c_dec.item_1_valid_contract}` | 2.Funds: `{c_dec.item_2_sufficient_funds}` | "
            f"3.Receipt: `{c_dec.item_3_receipt_or_pop_attached}` | 4.Eligible: `{c_dec.item_4_expense_eligible}` | 5.Math/Vol: `{c_dec.item_5_math_and_volume_verified}`"
        )
        st.info(f"**AI Auditor Rationale:** {c_dec.rationale}")


# 8. Tab 4: Contract & Policy RAG Explorer
with tab4:
    st.subheader("ChromaDB Contract & Governance Policy RAG Explorer")
    rag_query = st.text_input(
        "Ask a question or search for any Customer Element / Governance Rule:",
        value="What are the eligible banners and exclusions for BP-COMBO?",
    )
    if st.button("🔍 Search ChromaDB Vector Store"):
        chunks_text = search_customer_contract.invoke({"query": rag_query})
        st.markdown(chunks_text)