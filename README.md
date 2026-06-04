# Wallet Forensics Agent

An AI-powered forensic investigation tool for Ethereum wallet addresses. Combines deterministic on-chain risk analysis with a LangGraph agentic reasoning loop and LLM-generated forensic reports — designed to mirror the workflow used in real cryptocurrency crime investigations.


---

## Demo

> Enter any Ethereum wallet address → get a structured forensic risk report in under 30 seconds.

**Try these addresses:**
| Address | Expected Result |
|---------|----------------|
| `0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045` | Vitalik Buterin — LOW risk, high volume |
| `0x1da5821544e25c636c1417ba96ade4cf6d2f9b5a` | Lazarus Group (Ronin hack) — HIGH risk |
| `0x098b716b8aaf21512996dc57eb0615e2383e2f96` | Tornado Cash deposit — HIGH risk |

---

## How It Works

```
Wallet Address
      │
      ▼
┌─────────────┐
│  fetch_node │  Calls Etherscan V2 API
│             │  • ETH balance
│             │  • Last 100 transactions
│             │  • Last 100 ERC-20 transfers
│             │  • First-ever transaction (true wallet age)
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ analyze_node │  Deterministic risk engine (no LLM)
│              │  Extracts 7 signals → weighted score 0–100
└──────┬───────┘
       │
       ▼
  Risk Router
  (conditional)
  ┌────┴────┐
score < 50  score ≥ 50
  │              │
  │    ┌─────────▼──────────┐
  │    │   deep_dive_node   │  Fetches one-hop neighbour
  │    │                    │  transactions for top 3
  │    │                    │  flagged counterparties
  │    └─────────┬──────────┘
  │              │
  └──────┬───────┘
         ▼
  ┌─────────────┐
  │ report_node │  Groq LLaMA-3 generates structured
  │             │  forensic report with evidence citations
  └─────────────┘
         │
         ▼
  Streamlit UI
```

---

## Risk Signals

Seven deterministic signals are computed before any LLM is involved:

| Signal | What It Measures | Weight |
|--------|-----------------|--------|
| **Mixer Exposure** | Direct interaction with OFAC-sanctioned mixer contracts (Tornado Cash, Railgun, Sinbad) | 35 pts |
| **Fan-out Ratio** | Unique recipients ÷ total outgoing transactions — classic fund dispersal pattern | 20 pts |
| **Transaction Velocity** | Average transactions per day + 24-hour burst count | 15 pts |
| **Round-value Transactions** | Count of 0.1 / 1 / 10 / 100 ETH transactions — structuring indicator | 10 pts |
| **Account Age** | Days since first ever on-chain transaction (dedicated ascending-sort API call) | 10 pts |
| **Token Diversity** | Unique ERC-20 tokens interacted with — automated activity signal | 5 pts |
| **Incoming Only** | Wallet that never sends — possible collection or drop address | 5 pts |

**Scoring:** 0–39 = LOW, 40–69 = MEDIUM, 70–100 = HIGH

The mixer exposure signal carries the most weight because it is hard evidence — a contract address is either on the OFAC sanctions list or it isn't. All other signals are behavioural patterns.

---

## Forensic Report Structure

Every investigation produces a five-section LLM-generated report:

1. **Executive Summary** — What this wallet appears to be doing, overall verdict
2. **Risk Score** — Score with justification
3. **Risk Factors** — Each triggered flag explained forensically, with transaction hash citations
4. **Key Evidence** — The 3 most incriminating data points
5. **Recommended Actions** — Concrete investigator next steps (trace hashes, subpoena exchange, cluster analysis)

The LLM receives structured signal data and the 8 highest-value transactions — not a raw dump of all 100 — producing precise, citation-accurate output rather than vague summaries.

---

## Stack

| Component | Technology 
|-----------|-----------
| Agent framework | LangGraph
| LLM | Groq API (LLaMA-3.3-70b)
| Blockchain data | Etherscan V2 API
| UI | Streamlit |
| Hosting | Streamlit Cloud

---

## Project Structure

```
wallet-forensics-agent/
├── src/
│   ├── fetcher.py       # Etherscan V2 API wrapper (4 endpoints)
│   ├── risk_engine.py   # Deterministic risk signal extractor + scorer
│   ├── agent.py         # LangGraph StateGraph (4 nodes, conditional routing)
│   └── prompts.py       # LLM system prompt + report prompt builder
├── app.py               # Streamlit UI
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Design Decisions

**Why LangGraph over a simple chain?**
The conditional deep-dive branch requires genuine agentic decision-making — the system decides whether to fetch additional data based on intermediate results. A linear chain can't do this. LangGraph's `StateGraph` with conditional edges models this naturally and makes the routing logic explicit and testable.

**Why compute risk signals before calling the LLM?**
Passing structured signals (7 numbers + flag strings) to the LLM instead of raw transaction JSON produces dramatically better output. The model reasons over a forensic summary, not 100 rows of data. It also makes the risk score deterministic — the score is always the same for the same wallet, regardless of LLM temperature.

**Why a dedicated first-transaction API call?**
Fetching transactions with `sort=desc` and deriving age from the oldest transaction in the batch gives wrong results for old high-volume wallets. A 10-year-old wallet with thousands of transactions would show 22 days if the last 100 transactions only reach that far back. A separate `sort=asc, limit=1` call fetches the true first transaction regardless of total volume.

**Why is mixer exposure weighted at 35 points?**
Unlike behavioural signals that require interpretation, mixer contract addresses are either on the OFAC sanctions list or they aren't. A single Tornado Cash interaction is treated as hard evidence by financial intelligence units — the weighting reflects this.

---
