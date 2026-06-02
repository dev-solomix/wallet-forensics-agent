"""
app.py
------
Streamlit UI for the Crypto Wallet Forensics Agent.

Run with:
    streamlit run app.py

Layout:
  - Header + instructions
  - Wallet address input + Investigate button
  - Progress spinner while agent runs
  - Results: risk score badge | signal table | forensic report
  - Raw transactions expander at the bottom
"""

import time
import datetime
import streamlit as st

from src.agent import run_investigation


# ── page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Wallet Forensics Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Score badge */
  .score-badge {
      display: inline-block;
      padding: 0.45rem 1.1rem;
      border-radius: 8px;
      font-size: 2.4rem;
      font-weight: 700;
      letter-spacing: -1px;
      margin-bottom: 0.2rem;
  }
  .badge-low    { background: #d4edda; color: #155724; }
  .badge-medium { background: #fff3cd; color: #856404; }
  .badge-high   { background: #f8d7da; color: #721c24; }

  /* Signal table rows */
  .signal-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.4rem 0;
      border-bottom: 1px solid #f0f0f0;
      font-size: 0.9rem;
  }
  .signal-name  { color: #555; }
  .signal-value { font-weight: 600; color: #111; }
  .flag-dot     { width: 8px; height: 8px; border-radius: 50%;
                  display: inline-block; margin-right: 6px; }
  .dot-red    { background: #dc3545; }
  .dot-yellow { background: #ffc107; }
  .dot-green  { background: #28a745; }

  /* Section divider */
  .section-divider {
      border: none;
      border-top: 1px solid #eee;
      margin: 1.5rem 0;
  }

  /* Footer */
  .footer {
      text-align: center;
      font-size: 0.75rem;
      color: #aaa;
      margin-top: 3rem;
  }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def score_badge(score: int, label: str) -> str:
    css_class = {
        "LOW":    "badge-low",
        "MEDIUM": "badge-medium",
        "HIGH":   "badge-high",
    }.get(label, "badge-medium")
    return (
        f'<div class="score-badge {css_class}">'
        f'{score}<span style="font-size:1rem;font-weight:400;"> / 100</span>'
        f'</div>'
        f'<div style="font-size:1.1rem;font-weight:600;color:#333;">'
        f'{label} RISK</div>'
    )


def flag_dot(value, signal_name: str) -> str:
    """Return a coloured dot based on whether the signal is elevated."""
    red_signals = {"mixer_exposure", "incoming_only"}
    if signal_name in red_signals and value:
        return '<span class="flag-dot dot-red"></span>'
    if isinstance(value, float) and value > 0.7:
        return '<span class="flag-dot dot-red"></span>'
    if isinstance(value, (int, float)) and value > 0:
        return '<span class="flag-dot dot-yellow"></span>'
    return '<span class="flag-dot dot-green"></span>'


def format_signal_value(signal_name: str, signal: dict) -> str:
    val = signal.get("value")
    if signal_name == "transaction_velocity":
        return f"{val:.1f} txns/day"
    if signal_name == "fan_out_ratio":
        return f"{val:.0%} unique recipients"
    if signal_name == "mixer_exposure":
        return f"{val} flagged tx" + (" ⚠️" if val > 0 else "")
    if signal_name == "account_age_days":
        return f"{val:.0f} days old"
    if signal_name == "round_transactions":
        return f"{val} detected"
    if signal_name == "incoming_only":
        return "Yes ⚠️" if val else "No"
    if signal_name == "token_diversity":
        return f"{val} unique tokens"
    return str(val)


SIGNAL_LABELS = {
    "transaction_velocity": "Transaction Velocity",
    "fan_out_ratio":        "Fan-out Ratio",
    "mixer_exposure":       "Mixer Exposure",
    "account_age_days":     "Account Age",
    "round_transactions":   "Round-value Txns",
    "incoming_only":        "Incoming Only",
    "token_diversity":      "Token Diversity",
}


def ts_to_date(ts: int) -> str:
    if not ts:
        return "—"
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


# ── header ────────────────────────────────────────────────────────────────────

st.markdown("## 🔍 Crypto Wallet Forensics Agent")
st.markdown(
    "Investigates Ethereum wallet addresses using on-chain data analysis "
    "and AI-generated forensic reports. Powered by Etherscan + Groq (LLaMA-3)."
)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ── input area ────────────────────────────────────────────────────────────────

col_input, col_btn = st.columns([5, 1])

with col_input:
    address = st.text_input(
        label="Wallet Address",
        placeholder="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        label_visibility="collapsed",
    )

with col_btn:
    investigate = st.button("Investigate", type="primary", use_container_width=True)

# Example wallets
with st.expander("💡 Try an example wallet"):
    st.markdown("""
| Wallet | Description |
|--------|-------------|
| `0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045` | Vitalik Buterin — high volume, expected LOW risk |
| `0x1da5821544e25c636c1417ba96ade4cf6d2f9b5a` | Documented Lazarus Group address (Ronin hack) |
| `0x098b716b8aaf21512996dc57eb0615e2383e2f96` | Tornado Cash deposit address |
    """)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ── investigation logic ───────────────────────────────────────────────────────

if investigate and address:

    if not address.startswith("0x") or len(address) < 40:
        st.error("⚠️  That doesn't look like a valid Ethereum address. "
                 "It should start with 0x and be 42 characters long.")
        st.stop()

    # ── run the agent ─────────────────────────────────────────────────────────
    with st.spinner("Fetching on-chain data from Etherscan..."):
        start_time = time.time()
        result     = run_investigation(address)
        elapsed    = time.time() - start_time

    # ── fatal error ───────────────────────────────────────────────────────────
    if result.get("error"):
        st.error(f"**Investigation failed:** {result['error']}")
        st.stop()

    risk = result.get("risk_assessment", {})
    data = result.get("wallet_data",     {})

    if not risk:
        st.warning("No risk data returned. The wallet may have no transaction history.")
        st.stop()

    score   = risk.get("risk_score", 0)
    label   = risk.get("risk_label", "UNKNOWN")
    signals = risk.get("signals",    {})
    flags   = risk.get("top_flags",  [])
    meta    = risk.get("metadata",   {})
    report  = result.get("report",   "")
    routed  = "deep_dive" if score >= 50 else "direct"

    # ── results layout ────────────────────────────────────────────────────────
    st.success(f"Investigation complete in {elapsed:.1f}s")

    # Row 1: score | metadata | flags
    col_score, col_meta, col_flags = st.columns([1, 1.5, 2.5])

    with col_score:
        st.markdown("**Risk Score**")
        st.markdown(score_badge(score, label), unsafe_allow_html=True)
        route_label = "🔎 Deep-dive triggered" if routed == "deep_dive" else "✅ Standard route"
        st.caption(route_label)

    with col_meta:
        st.markdown("**Wallet Summary**")
        st.markdown(f"""
- **Address:** `{address[:10]}...{address[-6:]}`
- **Balance:** `{meta.get('eth_balance', 0):.4f} ETH`
- **Txns analysed:** `{meta.get('total_tx_analysed', 0)}`
        """)

    with col_flags:
        st.markdown("**Flags Raised**")
        for flag in flags:
            icon = "🔴" if any(w in flag.lower() for w in ["mixer", "tornado", "sanction"]) else "🟡"
            if "no significant" in flag.lower():
                icon = "🟢"
            st.markdown(f"{icon} {flag}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Row 2: signal breakdown | forensic report
    col_signals, col_report = st.columns([1, 2])

    with col_signals:
        st.markdown("**Signal Breakdown**")
        for sig_key, sig_label in SIGNAL_LABELS.items():
            if sig_key not in signals:
                continue
            sig      = signals[sig_key]
            dot_html = flag_dot(sig.get("value"), sig_key)
            val_str  = format_signal_value(sig_key, sig)
            st.markdown(
                f'<div class="signal-row">'
                f'<span class="signal-name">{dot_html}{sig_label}</span>'
                f'<span class="signal-value">{val_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Token breakdown if noteworthy
        token_div = signals.get("token_diversity", {})
        breakdown = token_div.get("token_breakdown", {})
        if breakdown and len(breakdown) >= 5:
            st.markdown("")
            with st.expander("Token breakdown"):
                for sym, count in list(breakdown.items())[:10]:
                    st.markdown(f"- `{sym}`: {count} transfer(s)")

    with col_report:
        st.markdown("**Forensic Report**")
        if report:
            st.markdown(report)
        else:
            st.info("No report generated.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── raw transactions expander ─────────────────────────────────────────────
    with st.expander("📋 View raw ETH transactions"):
        eth_txns = data.get("eth_transactions", [])
        if eth_txns:
            rows = []
            for tx in eth_txns:
                rows.append({
                    "Date":      ts_to_date(tx.get("timestamp", 0)),
                    "Direction": tx.get("direction", "").upper(),
                    "ETH":       f"{tx.get('value_eth', 0):.6f}",
                    "From":      tx.get("from_address", "")[:18] + "...",
                    "To":        tx.get("to_address",   "")[:18] + "...",
                    "Hash":      tx.get("hash", "")[:18] + "...",
                    "Error":     "Yes" if tx.get("is_error") else "No",
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No ETH transactions found.")

    with st.expander("🪙 View ERC-20 token transfers"):
        token_txns = data.get("token_transfers", [])
        if token_txns:
            rows = []
            for tx in token_txns:
                rows.append({
                    "Date":      ts_to_date(tx.get("timestamp", 0)),
                    "Direction": tx.get("direction", "").upper(),
                    "Token":     tx.get("token_symbol", "?"),
                    "Value":     f"{tx.get('value', 0):.4f}",
                    "From":      tx.get("from_address", "")[:18] + "...",
                    "To":        tx.get("to_address",   "")[:18] + "...",
                    "Hash":      tx.get("hash", "")[:18] + "...",
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No token transfers found.")

    # Deep dive results if triggered
    deep = result.get("deep_dive", {})
    neighbours = deep.get("neighbour_summaries", [])
    if neighbours:
        with st.expander("🕵️ Deep-dive: one-hop neighbour analysis"):
            st.caption(
                "For high-risk wallets, the agent fetches one additional hop "
                "of transaction data from the top flagged counterparties."
            )
            for nb in neighbours:
                if nb.get("error"):
                    st.markdown(f"- `{nb['address']}` — could not fetch data")
                    continue
                st.markdown(
                    f"**`{nb['address']}`** — "
                    f"{nb['tx_count']} txns fetched, "
                    f"{nb['unique_destinations']} unique onward destinations"
                )

elif investigate and not address:
    st.warning("Please enter a wallet address first.")


# ── idle state ────────────────────────────────────────────────────────────────

if not investigate:
    st.markdown("""
    **How it works:**

    1. Enter any Ethereum wallet address above
    2. The agent fetches the last 100 transactions from Etherscan
    3. Seven forensic risk signals are extracted and scored
    4. High-risk wallets (score ≥ 50) trigger a one-hop deep-dive into counterparty addresses
    5. A structured forensic report is generated by LLaMA-3 on Groq

    **Risk signals analysed:** transaction velocity · fan-out ratio · mixer exposure ·
    account age · round-value transactions · incoming-only pattern · token diversity
    """)


# ── footer ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="footer">Built with LangGraph · Groq (LLaMA-3) · Etherscan API · Streamlit'
    '<br>For educational and research purposes only.</div>',
    unsafe_allow_html=True,
)