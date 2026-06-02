"""
risk_engine.py
--------------
Extracts seven forensic risk signals from raw wallet data
produced by fetcher.py, then computes a weighted risk score (0–100).

No LLM involved. Pure deterministic logic.
This runs BEFORE the LLM so the report node gets structured
signals rather than raw JSON — cheaper tokens, better output.

Signals computed:
  1. transaction_velocity   — avg transactions per day
  2. fan_out_ratio          — unique recipients / total outgoing txns
  3. mixer_exposure         — interaction with known mixer contracts
  4. account_age_days       — days since first ever transaction
  5. round_number_txns      — count of suspiciously round ETH values
  6. incoming_only          — wallet never sends, only receives
  7. token_diversity        — unique ERC-20 tokens in short timeframe
"""

import time
from collections import Counter


# ── known mixer / tumbler contract addresses ──────────────────────────────────
# Sources: Chainalysis public reports, OFAC sanctions list (Aug 2022),
#          community-maintained lists. Lowercase, no checksums.

KNOWN_MIXER_ADDRESSES = {
    # Tornado Cash core contracts (OFAC sanctioned Aug 2022)
    "0x722122df12d4e14e13ac3b6895a86e84145b6967",  # TC Router
    "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",  # TC 0.1 ETH
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",  # TC 1 ETH
    "0xa160cdab225685da1d56aa342ad8841c3b53f291",  # TC 10 ETH
    "0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936",  # TC 100 ETH
    "0x12d66f87a04a9e220c9d8c4194f49a52afebf9e2",  # TC 0.1 ETH (alt)
    "0x1e34a77868e19a6647b1f2f47b51ed72dede95dd",  # TC 1 ETH   (alt)
    "0xd96f2b1c14db8458374d9aca76e26c3950113464",  # TC 10 ETH  (alt)
    "0x169ad27a470d064dede56a2d3ff727986b15d52b",  # TC 100 ETH (alt)
    "0x0836222f2b2b5a6fc140c537a254a67f7bda4094",  # TC 1000 ETH
    # Railgun privacy contracts
    "0xfa7093cdd9ee6932b4eb2c9e1cde7ce00b1fa4b5",  # Railgun v1
    "0xbd6015b34bcbf789e3d0a9abb26e2f80cb37e5b1",  # Railgun v2
    # eXch (unlicensed exchange flagged for sanctions evasion)
    "0x6226e00bcaf8462735f7bfd2884af57a6e6f966a",
    # Sinbad (OFAC sanctioned Nov 2023)
    "0x085fe6e8fc1d851d815ba7e77e7d66b8ab59b42a",
}

# Round ETH values that commonly appear in structuring / mixing
ROUND_VALUE_THRESHOLDS = {0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0}
ROUND_VALUE_TOLERANCE  = 0.0001   # within 0.01% counts as "round"

# Thresholds for score weighting (tuned against known-bad wallets)
VELOCITY_HIGH        = 20    # txns/day above this = high risk
VELOCITY_MEDIUM      = 5     # txns/day above this = medium risk
FAN_OUT_HIGH         = 0.90  # 90%+ unique recipients = high risk
FAN_OUT_MEDIUM       = 0.70
ACCOUNT_AGE_NEW      = 7     # days — very new wallet
ACCOUNT_AGE_MEDIUM   = 30    # days
ROUND_TXN_HIGH       = 5     # 5+ round-value txns
ROUND_TXN_MEDIUM     = 2
TOKEN_DIVERSITY_HIGH = 15    # unique tokens
TOKEN_DIVERSITY_MED  = 8


# ── individual signal extractors ──────────────────────────────────────────────

def _compute_velocity(eth_txns: list, token_txns: list) -> dict:
    """
    Transactions per day, averaged across the wallet's active lifespan.
    A wallet doing 20+ txns/day warrants scrutiny.
    """
    all_txns = eth_txns + token_txns
    if len(all_txns) < 2:
        return {"value": 0.0, "tx_count": len(all_txns), "lifespan_days": 0}

    timestamps = [tx["timestamp"] for tx in all_txns if tx["timestamp"] > 0]
    if not timestamps:
        return {"value": 0.0, "tx_count": 0, "lifespan_days": 0}

    oldest    = min(timestamps)
    newest    = max(timestamps)
    span_secs = newest - oldest

    # Avoid division by zero for wallets with all txns in same block
    lifespan_days = max(span_secs / 86400, 1)
    velocity      = len(all_txns) / lifespan_days

    return {
        "value":         round(velocity, 2),
        "tx_count":      len(all_txns),
        "lifespan_days": round(lifespan_days, 1),
    }


def _compute_fan_out(eth_txns: list) -> dict:
    """
    Fan-out ratio = unique outgoing recipients / total outgoing txns.
    Ratio close to 1.0 means every send goes to a different address —
    a classic fund-dispersal or structuring pattern.
    """
    outgoing = [tx for tx in eth_txns if tx.get("direction") == "out"]
    if not outgoing:
        return {"value": 0.0, "unique_recipients": 0, "total_outgoing": 0}

    recipients        = [tx["to_address"] for tx in outgoing]
    unique_recipients = len(set(recipients))
    ratio             = unique_recipients / len(outgoing)

    return {
        "value":             round(ratio, 4),
        "unique_recipients": unique_recipients,
        "total_outgoing":    len(outgoing),
    }


def _compute_mixer_exposure(eth_txns: list, token_txns: list) -> dict:
    """
    Check whether any counterparty address is a known mixer contract.
    Checks both from_address and to_address on every transaction.
    """
    flagged_txns   = []
    flagged_addrs  = set()

    for tx in eth_txns + token_txns:
        hit = None
        if tx.get("from_address") in KNOWN_MIXER_ADDRESSES:
            hit = tx["from_address"]
        elif tx.get("to_address") in KNOWN_MIXER_ADDRESSES:
            hit = tx["to_address"]

        if hit:
            flagged_addrs.add(hit)
            flagged_txns.append({
                "hash":            tx.get("hash", ""),
                "mixer_address":   hit,
                "direction":       tx.get("direction", ""),
            })

    return {
        "value":          len(flagged_txns),        # 0 = clean
        "flagged_txns":   flagged_txns[:10],        # cap for LLM context
        "flagged_addrs":  list(flagged_addrs),
        "is_exposed":     len(flagged_txns) > 0,
    }


def _compute_account_age(eth_txns: list, token_txns: list) -> dict:
    """
    Days between the wallet's first ever transaction and today.
    Wallets under 7 days old moving meaningful value are suspicious.
    """
    all_txns   = eth_txns + token_txns
    timestamps = [tx["timestamp"] for tx in all_txns if tx["timestamp"] > 0]

    if not timestamps:
        return {"value": 0, "first_seen_ts": None}

    first_seen    = min(timestamps)
    age_days      = (time.time() - first_seen) / 86400

    return {
        "value":        round(age_days, 1),
        "first_seen_ts": first_seen,
    }


def _compute_round_transactions(eth_txns: list) -> dict:
    """
    Count transactions with suspiciously round ETH values.
    Humans and scripts structuring funds often use clean round numbers.
    """
    round_txns = []

    for tx in eth_txns:
        value = tx.get("value_eth", 0.0)
        if value <= 0:
            continue
        for threshold in ROUND_VALUE_THRESHOLDS:
            if abs(value - threshold) <= ROUND_VALUE_TOLERANCE:
                round_txns.append({
                    "hash":      tx.get("hash", ""),
                    "value_eth": value,
                    "direction": tx.get("direction", ""),
                })
                break   # only count each tx once

    return {
        "value":      len(round_txns),
        "round_txns": round_txns[:10],  # cap for LLM context
    }


def _compute_incoming_only(eth_txns: list, token_txns: list) -> dict:
    """
    A wallet that only ever receives and never sends is either a
    cold storage address, a collection address for a scam, or
    a drop wallet used in layering.
    """
    all_txns     = eth_txns + token_txns
    has_outgoing = any(tx.get("direction") == "out" for tx in all_txns)
    has_incoming = any(tx.get("direction") == "in"  for tx in all_txns)

    return {
        "value":        not has_outgoing and has_incoming,  # True = flag
        "has_outgoing": has_outgoing,
        "has_incoming": has_incoming,
    }


def _compute_token_diversity(token_txns: list) -> dict:
    """
    Count unique ERC-20 tokens the wallet has interacted with.
    Very high token diversity in a short window can indicate
    automated activity, airdrop farming, or wash trading.
    """
    symbols  = [tx.get("token_symbol", "UNKNOWN") for tx in token_txns]
    counts   = Counter(symbols)
    unique   = len(counts)

    return {
        "value":          unique,
        "token_breakdown": dict(counts.most_common(10)),
    }


# ── scorer ────────────────────────────────────────────────────────────────────

def _compute_score(signals: dict) -> int:
    """
    Weighted risk score from 0 (clean) to 100 (high risk).

    Weight allocation (totals to 100):
      mixer_exposure      35  — hard evidence, heaviest weight
      fan_out_ratio       20  — structural pattern
      transaction_velocity 15 — behavioural pattern
      round_transactions  10  — structuring indicator
      account_age         10  — new wallet penalty
      token_diversity      5  — supporting signal
      incoming_only        5  — supporting signal
    """
    score = 0

    # ── mixer exposure (0–35) ─────────────────────────────────────────────────
    mixer = signals["mixer_exposure"]["value"]
    if mixer >= 3:
        score += 35
    elif mixer >= 1:
        score += 20

    # ── fan-out ratio (0–20) ──────────────────────────────────────────────────
    fan_out = signals["fan_out_ratio"]["value"]
    if fan_out >= FAN_OUT_HIGH:
        score += 20
    elif fan_out >= FAN_OUT_MEDIUM:
        score += 10

    # ── transaction velocity (0–15) ───────────────────────────────────────────
    velocity = signals["transaction_velocity"]["value"]
    if velocity >= VELOCITY_HIGH:
        score += 15
    elif velocity >= VELOCITY_MEDIUM:
        score += 7

    # ── round transactions (0–10) ─────────────────────────────────────────────
    round_count = signals["round_transactions"]["value"]
    if round_count >= ROUND_TXN_HIGH:
        score += 10
    elif round_count >= ROUND_TXN_MEDIUM:
        score += 5

    # ── account age (0–10) ────────────────────────────────────────────────────
    age = signals["account_age_days"]["value"]
    if 0 < age <= ACCOUNT_AGE_NEW:
        score += 10
    elif age <= ACCOUNT_AGE_MEDIUM:
        score += 5

    # ── token diversity (0–5) ─────────────────────────────────────────────────
    diversity = signals["token_diversity"]["value"]
    if diversity >= TOKEN_DIVERSITY_HIGH:
        score += 5
    elif diversity >= TOKEN_DIVERSITY_MED:
        score += 2

    # ── incoming only (0–5) ───────────────────────────────────────────────────
    if signals["incoming_only"]["value"]:
        score += 5

    return min(score, 100)   # cap at 100


def _risk_label(score: int) -> str:
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    return "LOW"


# ── public API ────────────────────────────────────────────────────────────────

def extract_risk_signals(wallet_data: dict) -> dict:
    """
    Master function. Takes the dict returned by fetch_wallet_data()
    and returns a complete risk assessment.

    This is the only function agent.py needs to import from this module.

    Returns:
        {
          "address":              str,
          "risk_score":           int,       # 0–100
          "risk_label":           str,       # LOW / MEDIUM / HIGH
          "signals": {
            "transaction_velocity": {...},
            "fan_out_ratio":        {...},
            "mixer_exposure":       {...},
            "account_age_days":     {...},
            "round_transactions":   {...},
            "incoming_only":        {...},
            "token_diversity":      {...},
          },
          "top_flags":            list[str], # human-readable findings
          "metadata": {
            "eth_balance":         float,
            "total_tx_analysed":   int,
          }
        }
    """
    eth_txns   = wallet_data.get("eth_transactions", [])
    token_txns = wallet_data.get("token_transfers",  [])
    address    = wallet_data.get("address", "unknown")

    signals = {
        "transaction_velocity": _compute_velocity(eth_txns, token_txns),
        "fan_out_ratio":        _compute_fan_out(eth_txns),
        "mixer_exposure":       _compute_mixer_exposure(eth_txns, token_txns),
        "account_age_days":     _compute_account_age(eth_txns, token_txns),
        "round_transactions":   _compute_round_transactions(eth_txns),
        "incoming_only":        _compute_incoming_only(eth_txns, token_txns),
        "token_diversity":      _compute_token_diversity(token_txns),
    }

    score = _compute_score(signals)
    label = _risk_label(score)

    # Build a plain-English list of triggered flags for the LLM prompt
    top_flags = _build_flags(signals, score)

    return {
        "address":    address,
        "risk_score": score,
        "risk_label": label,
        "signals":    signals,
        "top_flags":  top_flags,
        "metadata": {
            "eth_balance":       wallet_data.get("balance_eth", 0.0),
            "total_tx_analysed": wallet_data.get("total_tx_count", 0),
        },
    }


def _build_flags(signals: dict, score: int) -> list:
    """
    Convert triggered signal thresholds into plain-English flag strings.
    These are passed directly into the LLM forensic prompt.
    """
    flags = []

    mixer = signals["mixer_exposure"]
    if mixer["is_exposed"]:
        addrs = ", ".join(mixer["flagged_addrs"][:3])
        flags.append(
            f"Direct interaction with {len(mixer['flagged_txns'])} known "
            f"mixer transaction(s). Addresses: {addrs}"
        )

    fan_out = signals["fan_out_ratio"]
    if fan_out["value"] >= FAN_OUT_MEDIUM:
        flags.append(
            f"High fan-out ratio: {fan_out['value']:.0%} of outgoing "
            f"transactions went to unique addresses "
            f"({fan_out['unique_recipients']} of {fan_out['total_outgoing']})"
        )

    velocity = signals["transaction_velocity"]
    if velocity["value"] >= VELOCITY_MEDIUM:
        flags.append(
            f"Elevated transaction velocity: {velocity['value']:.1f} "
            f"transactions/day over {velocity['lifespan_days']} days"
        )

    age = signals["account_age_days"]
    if 0 < age["value"] <= ACCOUNT_AGE_MEDIUM:
        flags.append(
            f"Young wallet: first seen {age['value']:.0f} days ago"
        )

    rounds = signals["round_transactions"]
    if rounds["value"] >= ROUND_TXN_MEDIUM:
        flags.append(
            f"{rounds['value']} transaction(s) with suspiciously round "
            f"ETH values (0.1, 1, 10, 100 ETH etc.)"
        )

    diversity = signals["token_diversity"]
    if diversity["value"] >= TOKEN_DIVERSITY_MED:
        flags.append(
            f"High token diversity: {diversity['value']} unique ERC-20 "
            f"tokens interacted with"
        )

    if signals["incoming_only"]["value"]:
        flags.append(
            "Incoming-only wallet: no outgoing transactions observed "
            "— possible collection or drop address"
        )

    if not flags:
        flags.append("No significant risk indicators detected.")

    return flags


# ── quick test ────────────────────────────────────────────────────────────────
# Runs without an API call — uses synthetic data so you can test the
# scoring logic without needing a live Etherscan key.
#
#   python src/risk_engine.py

if __name__ == "__main__":

    print("\nRisk engine — synthetic data test")
    print("-" * 60)

    # Simulate a suspicious wallet:
    # - 80 outgoing txns, all to different addresses (high fan-out)
    # - several round-number ETH values
    # - interaction with a Tornado Cash address
    # - wallet only 3 days old

    now = int(time.time())

    synthetic_eth_txns = []

    # 80 outgoing to unique addresses
    for i in range(80):
        synthetic_eth_txns.append({
            "hash":         f"0xabc{i:04x}",
            "timestamp":    now - (i * 300),          # every 5 min
            "from_address": "0xtargetwalletaddress",
            "to_address":   f"0xunique_recipient_{i:04x}",
            "value_eth":    1.0 if i % 10 == 0 else 0.037,  # some round
            "direction":    "out",
            "block_number": 19000000 - i,
            "is_error":     False,
            "gas_used":     21000,
        })

    # 1 incoming from Tornado Cash
    synthetic_eth_txns.append({
        "hash":         "0xmixerhash001",
        "timestamp":    now - 86400,
        "from_address": "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",  # TC 1 ETH
        "to_address":   "0xtargetwalletaddress",
        "value_eth":    10.0,
        "direction":    "in",
        "block_number": 18999000,
        "is_error":     False,
        "gas_used":     21000,
    })

    synthetic_token_txns = [
        {
            "hash":             f"0xtoken{i:03x}",
            "timestamp":        now - (i * 600),
            "from_address":     "0xtargetwalletaddress",
            "to_address":       f"0xtoken_recipient_{i}",
            "token_symbol":     f"TOKEN{i}",
            "token_name":       f"Test Token {i}",
            "value":            100.0,
            "direction":        "out",
            "contract_address": f"0xcontract{i}",
        }
        for i in range(20)   # 20 unique tokens
    ]

    synthetic_wallet = {
        "address":          "0xtargetwalletaddress",
        "balance_eth":      4.2,
        "eth_transactions": synthetic_eth_txns,
        "token_transfers":  synthetic_token_txns,
        "fetch_errors":     [],
        "total_tx_count":   len(synthetic_eth_txns) + len(synthetic_token_txns),
    }

    result = extract_risk_signals(synthetic_wallet)

    print(f"  Address:      {result['address']}")
    print(f"  Risk score:   {result['risk_score']} / 100")
    print(f"  Risk label:   {result['risk_label']}")
    print(f"  ETH balance:  {result['metadata']['eth_balance']} ETH")
    print(f"  Txns analysed:{result['metadata']['total_tx_analysed']}")

    print("\n  Signals:")
    for name, sig in result["signals"].items():
        print(f"    {name:<28} {sig['value']}")

    print("\n  Flags raised:")
    for flag in result["top_flags"]:
        print(f"    - {flag}")

    expected_high = result["risk_label"] == "HIGH"
    print(f"\n  [{'OK' if expected_high else 'FAIL'}] "
          f"Expected HIGH risk — got {result['risk_label']}\n")