"""
agent.py
--------
LangGraph agent for crypto wallet forensic investigation.

Graph structure:
                    ┌─────────────┐
                    │  fetch_node │  ← calls Etherscan API
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │analyze_node │  ← runs risk_engine (no LLM)
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │      risk_router        │  ← conditional edge
              └────┬──────────────┬─────┘
              score < 50        score >= 50
                   │                  │
          ┌────────▼───────┐  ┌───────▼────────┐
          │  report_node   │  │ deep_dive_node  │  ← one-hop neighbour fetch
          └────────────────┘  └───────┬─────────┘
                                      │
                             ┌────────▼───────┐
                             │  report_node   │  ← Groq LLM call
                             └────────────────┘
"""

import os
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.fetcher import fetch_wallet_data, fetch_eth_transactions
from src.risk_engine import extract_risk_signals
from src.prompts import SYSTEM_PROMPT, build_report_prompt

load_dotenv()


# ── agent state ───────────────────────────────────────────────────────────────

class ForensicState(TypedDict):
    """
    The single state object that flows through every node.
    Each node reads from it and writes back to it.
    LangGraph merges return dicts — only return keys you're updating.
    """
    address:         str          # wallet address under investigation
    wallet_data:     dict         # raw output from fetcher.py
    risk_assessment: dict         # scored output from risk_engine.py
    deep_dive:       dict         # one-hop neighbour data (HIGH risk only)
    report:          str          # final LLM-generated forensic report
    error:           str          # any fatal error message


# ── node 1: fetch ─────────────────────────────────────────────────────────────

def fetch_node(state: ForensicState) -> dict:
    """
    Calls the Etherscan fetcher and writes raw wallet data to state.
    Isolated here so if the API fails, the error is caught cleanly
    and the graph can terminate gracefully.
    """
    try:
        wallet_data = fetch_wallet_data(state["address"])
        return {"wallet_data": wallet_data, "error": ""}
    except RuntimeError as exc:
        return {
            "wallet_data": {},
            "error": f"Fetch failed: {exc}",
        }


# ── node 2: analyze ───────────────────────────────────────────────────────────

def analyze_node(state: ForensicState) -> dict:
    """
    Runs the deterministic risk engine on the fetched wallet data.
    No API calls — pure Python computation.
    Skips gracefully if fetch_node reported an error.
    """
    if state.get("error"):
        return {"risk_assessment": {}}

    if not state.get("wallet_data"):
        return {
            "risk_assessment": {},
            "error": "No wallet data to analyse.",
        }

    try:
        risk_assessment = extract_risk_signals(state["wallet_data"])
        return {"risk_assessment": risk_assessment}
    except Exception as exc:
        return {
            "risk_assessment": {},
            "error": f"Analysis failed: {exc}",
        }


# ── conditional edge: risk router ─────────────────────────────────────────────

def risk_router(state: ForensicState) -> str:
    """
    Decides the next node based on risk score.
    Returns a string key that LangGraph maps to the next node.

      score >= 50  →  "deep_dive"   (investigate further before reporting)
      score <  50  →  "report"      (go straight to report generation)
      any error    →  "report"      (report will surface the error)
    """
    if state.get("error"):
        return "report"

    score = state.get("risk_assessment", {}).get("risk_score", 0)
    return "deep_dive" if score >= 50 else "report"


# ── node 3: deep dive (high-risk only) ───────────────────────────────────────

def deep_dive_node(state: ForensicState) -> dict:
    """
    For HIGH-risk wallets only.
    Takes the top 3 flagged counterparty addresses and fetches
    one additional hop of their transaction data — i.e. where
    did the money go NEXT after leaving the target wallet.

    This is what separates a basic wallet checker from a real
    investigative tool: it follows the money one step further.
    """
    deep_dive = {
        "neighbour_summaries": [],
        "error": None,
    }

    risk_assessment = state.get("risk_assessment", {})
    signals         = risk_assessment.get("signals", {})
    eth_txns        = state.get("wallet_data", {}).get("eth_transactions", [])

    # Collect the top 3 outgoing counterparties by value
    outgoing = [tx for tx in eth_txns if tx.get("direction") == "out"]
    outgoing_sorted = sorted(
        outgoing, key=lambda x: x.get("value_eth", 0), reverse=True
    )

    # Also include any mixer-flagged addresses
    mixer_addrs = set(
        signals.get("mixer_exposure", {}).get("flagged_addrs", [])
    )

    # Combine: top value recipients + mixer addresses, deduplicated, max 3
    priority_addrs = []
    seen = set()
    for tx in outgoing_sorted:
        addr = tx.get("to_address", "")
        if addr and addr not in seen:
            priority_addrs.append(addr)
            seen.add(addr)
        if len(priority_addrs) >= 3:
            break

    # Prepend mixer addresses (highest investigative priority)
    for addr in mixer_addrs:
        if addr not in seen:
            priority_addrs.insert(0, addr)
            seen.add(addr)

    priority_addrs = priority_addrs[:3]   # hard cap at 3 (API rate limit)

    for neighbour_addr in priority_addrs:
        try:
            neighbour_txns = fetch_eth_transactions(neighbour_addr, limit=20)

            # Summarise where this address sent funds next
            onward = [
                tx["to_address"]
                for tx in neighbour_txns
                if tx.get("direction") == "out"
            ]
            unique_destinations = len(set(onward))

            deep_dive["neighbour_summaries"].append({
                "address":             neighbour_addr,
                "tx_count":            len(neighbour_txns),
                "unique_destinations": unique_destinations,
                "sample_txns": [
                    {
                        "hash":      tx["hash"],
                        "value_eth": tx["value_eth"],
                        "direction": tx["direction"],
                    }
                    for tx in neighbour_txns[:3]
                ],
            })
        except RuntimeError:
            # Non-fatal: if a neighbour fetch fails, skip it and continue
            deep_dive["neighbour_summaries"].append({
                "address": neighbour_addr,
                "error":   "Could not fetch neighbour data",
            })

    return {"deep_dive": deep_dive}


# ── node 4: report ────────────────────────────────────────────────────────────

def report_node(state: ForensicState) -> dict:
    """
    Calls the Groq LLM to generate the forensic report.
    Runs for both LOW/MEDIUM wallets (no deep dive) and
    HIGH-risk wallets (after deep_dive_node has run).
    """
    # Surface any upstream errors cleanly in the report
    if state.get("error"):
        return {
            "report": f"## Investigation Error\n\n{state['error']}\n\n"
                      f"Please check the wallet address and API key, then retry."
        }

    risk_assessment = state.get("risk_assessment", {})
    wallet_data     = state.get("wallet_data", {})
    deep_dive       = state.get("deep_dive")

    if not risk_assessment:
        return {"report": "## Error\n\nNo risk assessment data available."}

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return {
            "report": "## Configuration Error\n\n"
                      "GROQ_API_KEY not found in .env. "
                      "Add it and restart."
        }

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,       # low temp = consistent, precise output
            max_tokens=1500,
            api_key=groq_api_key,
        )

        user_prompt = build_report_prompt(
            risk_assessment=risk_assessment,
            wallet_data=wallet_data,
            deep_dive=deep_dive,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        report   = response.content

    except Exception as exc:
        report = (
            f"## LLM Error\n\n"
            f"Report generation failed: {exc}\n\n"
            f"Risk score is still available: "
            f"{risk_assessment.get('risk_score', 'N/A')} / 100 "
            f"({risk_assessment.get('risk_label', 'N/A')})"
        )

    return {"report": report}


# ── graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    """
    Assembles and compiles the LangGraph StateGraph.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(ForensicState)

    # Register nodes
    graph.add_node("fetch",     fetch_node)
    graph.add_node("analyze",   analyze_node)
    graph.add_node("deep_dive", deep_dive_node)
    graph.add_node("report",    report_node)

    # Entry point
    graph.set_entry_point("fetch")

    # Linear edges
    graph.add_edge("fetch",   "analyze")

    # Conditional edge after analyze
    graph.add_conditional_edges(
        "analyze",
        risk_router,
        {
            "deep_dive": "deep_dive",
            "report":    "report",
        }
    )

    # Deep dive always flows into report
    graph.add_edge("deep_dive", "report")

    # Report is the terminal node
    graph.add_edge("report", END)

    return graph.compile()


# ── public API ────────────────────────────────────────────────────────────────

# Compile once at import time — reused across all Streamlit calls
forensic_agent = build_graph()


def run_investigation(address: str) -> dict:
    """
    Single entry point for app.py and any other caller.

    Args:
        address: Ethereum wallet address (0x...)

    Returns the final ForensicState dict containing:
        - risk_assessment  (score, label, signals, flags)
        - report           (LLM-generated markdown)
        - wallet_data      (raw transactions)
        - error            (empty string if successful)
    """
    initial_state: ForensicState = {
        "address":         address.strip(),
        "wallet_data":     {},
        "risk_assessment": {},
        "deep_dive":       {},
        "report":          "",
        "error":           "",
    }

    return forensic_agent.invoke(initial_state)


# ── quick test (no API keys needed for graph structure test) ──────────────────
# Tests that the graph compiles and routes correctly using mock nodes.
#
#   python src/agent.py

if __name__ == "__main__":
    print("\nAgent graph structure test")
    print("-" * 60)

    # Verify graph compiled
    graph = build_graph()
    print("  [OK] Graph compiled successfully")

    # Verify nodes are registered
    expected_nodes = {"fetch", "analyze", "deep_dive", "report"}
    actual_nodes   = set(graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
    missing        = expected_nodes - actual_nodes

    if missing:
        print(f"  [FAIL] Missing nodes: {missing}")
    else:
        print(f"  [OK] All nodes registered: {sorted(actual_nodes)}")

    # Verify conditional routing logic directly
    mock_high = {"risk_assessment": {"risk_score": 75}, "error": ""}
    mock_low  = {"risk_assessment": {"risk_score": 30}, "error": ""}
    mock_err  = {"risk_assessment": {},                  "error": "API failed"}

    assert risk_router(mock_high) == "deep_dive", "HIGH score should route to deep_dive"
    assert risk_router(mock_low)  == "report",    "LOW score should route to report"
    assert risk_router(mock_err)  == "report",    "Errors should route to report"
    print("  [OK] Conditional routing correct (high→deep_dive, low→report, error→report)")

    print("\n  Graph is ready. To run a live investigation:")
    print("  >>> from src.agent import run_investigation")
    print("  >>> result = run_investigation('0xYourAddressHere')")
    print("  >>> print(result['report'])\n")