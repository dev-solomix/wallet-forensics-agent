"""
prompts.py
----------
All LLM prompt templates for the forensic report node.
"""

SYSTEM_PROMPT = """You are a senior blockchain forensic analyst supporting \
law enforcement and financial crime investigations. Your job is to produce \
clear, evidence-based risk reports on Ethereum wallet addresses.

Rules you must follow:
- Cite specific transaction hashes as evidence wherever available.
- Never speculate beyond what the on-chain data supports.
- Use precise, professional language — your reports may be used in legal proceedings.
- Always structure your output using the exact five-section format requested.
- If a signal is absent, state that explicitly rather than omitting it."""


def build_report_prompt(risk_assessment: dict, wallet_data: dict,
                        deep_dive: dict = None) -> str:
    """
    Builds the user-turn prompt sent to the LLM report node.
    """
    address  = risk_assessment["address"]
    score    = risk_assessment["risk_score"]
    label    = risk_assessment["risk_label"]
    flags    = risk_assessment["top_flags"]
    signals  = risk_assessment["signals"]
    balance  = risk_assessment["metadata"]["eth_balance"]
    tx_count = risk_assessment["metadata"]["total_tx_analysed"]

    # Top 8 highest-value ETH transactions for evidence
    eth_txns = sorted(
        wallet_data.get("eth_transactions", []),
        key=lambda x: x.get("value_eth", 0),
        reverse=True
    )[:8]

    tx_lines = []
    for tx in eth_txns:
        tx_lines.append(
            f"  {tx['direction'].upper():3}  "
            f"{tx['value_eth']:.4f} ETH  "
            f"{'to' if tx['direction'] == 'out' else 'from'}: "
            f"{tx['to_address'] if tx['direction'] == 'out' else tx['from_address']}  "
            f"hash: {tx['hash'][:20]}..."
        )
    tx_block = "\n".join(tx_lines) if tx_lines else "  No ETH transactions found."

    mixer_txns  = signals["mixer_exposure"].get("flagged_txns", [])
    mixer_lines = []
    for mx in mixer_txns[:5]:
        mixer_lines.append(
            f"  {mx['direction'].upper():3}  "
            f"mixer: {mx['mixer_address']}  "
            f"hash: {mx['hash'][:20]}..."
        )
    mixer_block = (
        "\n".join(mixer_lines) if mixer_lines else "  None detected."
    )

    deep_dive_block = ""
    if deep_dive and deep_dive.get("neighbour_summaries"):
        lines = ["\nONE-HOP NEIGHBOUR ANALYSIS (top flagged counterparties):"]
        for nb in deep_dive["neighbour_summaries"]:
            lines.append(
                f"  Address: {nb['address']}\n"
                f"  Txns fetched: {nb['tx_count']}\n"
                f"  Onward destinations: {nb['unique_destinations']}\n"
            )
        deep_dive_block = "\n".join(lines)

    # Use the display-formatted age (e.g. "10.5 years" or "22 days")
    age_display = signals["account_age_days"].get(
        "display", f"{signals['account_age_days']['value']:.0f} days"
    )

    prompt = f"""Produce a forensic risk report for the following wallet.

WALLET SUMMARY
--------------
Address:          {address}
ETH Balance:      {balance:.4f} ETH
Transactions:     {tx_count} analysed
Risk Score:       {score} / 100
Risk Label:       {label}

RISK SIGNALS
------------
Transaction velocity:  {signals['transaction_velocity']['value']} txns/day
Fan-out ratio:         {signals['fan_out_ratio']['value']:.2%} unique recipients
Mixer interactions:    {signals['mixer_exposure']['value']} flagged transactions
Account age:           {age_display}
Round-value txns:      {signals['round_transactions']['value']} detected
Incoming only:         {signals['incoming_only']['value']}
Token diversity:       {signals['token_diversity']['value']} unique ERC-20s

FLAGS RAISED
------------
{chr(10).join(f"- {f}" for f in flags)}

HIGHEST-VALUE ETH TRANSACTIONS (evidence)
------------------------------------------
{tx_block}

MIXER / SANCTIONS INTERACTIONS
-------------------------------
{mixer_block}
{deep_dive_block}

REQUIRED OUTPUT FORMAT
----------------------
Produce exactly these five sections with these exact headings:

## Executive Summary
2-3 sentences. State what this wallet appears to be doing and the \
overall risk verdict.

## Risk Score
State the score (e.g. 74/100 — HIGH) and explain in 2 sentences \
why it landed at that level.

## Risk Factors
List each flag that fired. For each one: name it, explain what it \
means forensically, and cite a specific transaction hash if available.

## Key Evidence
List the 3 most incriminating data points with transaction hashes.

## Recommended Actions
List 3-5 concrete next steps an investigator should take \
(e.g. trace specific hashes, subpoena exchange, cluster analysis).
"""
    return prompt