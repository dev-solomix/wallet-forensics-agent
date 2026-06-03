"""
risk_engine.py
--------------
Extracts seven forensic risk signals from raw wallet data
produced by fetcher.py, then computes a weighted risk score (0-100).

No LLM involved. Pure deterministic logic.

Signals computed:
  1. transaction_velocity   — avg transactions per day
  2. fan_out_ratio          — unique recipients / total outgoing txns
  3. mixer_exposure         — interaction with known mixer contracts
  4. account_age_days       — days since first ever transaction
  5. round_number_txns      — count of suspiciously round ETH values
  6. incoming_only          — wallet never sends, only receives
  7. token_diversity        — unique ERC-20 tokens interacted with
"""

import time
from collections import Counter


# ── known mixer / tumbler contract addresses ──────────────────────────────────

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

ROUND_VALUE_THRESHOLDS = {0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0}
ROUND_VALUE_TOLERANCE  = 0.0001

VELOCITY_HIGH        = 20
VELOCITY_MEDIUM      = 5
FAN_OUT_HIGH         = 0.90
FAN_OUT_MEDIUM       = 0.70
ACCOUNT_AGE_NEW      = 7
ACCOUNT_AGE_MEDIUM   = 30
ROUND_TXN_HIGH       = 5
ROUND_TXN_MEDIUM     = 2
TOKEN_DIVERSITY_HIGH = 15
TOKEN_DIVERSITY_MED  = 8


# ── FIX 7: age display helper (was missing entirely) ─────────────────────────

def format_age(age_days: float) -> str:
    """
    Returns a human-readable age string.
    Shows years when age >= 365 days, days otherwise.

    Examples:
        3835.0 days  →  "10.5 years"
        22.0 days    →  "22 days"
    """
    if age_days <= 0:
        return "Unknown"
    if age_days >= 365:
        years = age_days / 365.25
        return f"{years:.1f} years"
    return f"{int(age_days)} days"


# ── individual signal extractors ──────────────────────────────────────────────

def _compute_velocity(eth_txns: list, token_txns: list) -> dict:
    all_txns = eth_txns + token_txns
    if len(all_txns) < 2:
        return {"value": 0.0, "tx_count": len(all_txns), "lifespan_days": 0}

    timestamps = [tx["timestamp"] for tx in all_txns if tx["timestamp"] > 0]
    if not timestamps:
        return {"value": 0.0, "tx_count": 0, "lifespan_days": 0}

    oldest        = min(timestamps)
    newest        = max(timestamps)
    lifespan_days = max((newest - oldest) / 86400, 1)
    velocity      = len(all_txns) / lifespan_days

    return {
        "value":         round(velocity, 2),
        "tx_count":      len(all_txns),
        "lifespan_days": round(lifespan_days, 1),
    }


def _compute_fan_out(eth_txns: list) -> dict:
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
    flagged_txns  = []
    flagged_addrs = set()

    for tx in eth_txns + token_txns:
        hit = None
        if tx.get("from_address") in KNOWN_MIXER_ADDRESSES:
            hit = tx["from_address"]
        elif tx.get("to_address") in KNOWN_MIXER_ADDRESSES:
            hit = tx["to_address"]

        if hit:
            flagged_addrs.add(hit)
            flagged_txns.append({
                "hash":          tx.get("hash", ""),
                "mixer_address": hit,
                "direction":     tx.get("direction", ""),
            })

    return {
        "value":         len(flagged_txns),
        "flagged_txns":  flagged_txns[:10],
        "flagged_addrs": list(flagged_addrs),
        "is_exposed":    len(flagged_txns) > 0,
    }


# FIX 8: _compute_account_age now accepts first_tx_timestamp.
# The old version only used min(timestamps) of the fetched batch,
# which silently gave wrong results for old high-volume wallets.
def _compute_account_age(eth_txns: list, token_txns: list,
                         first_tx_timestamp: int = 0) -> dict:
    """
    Days since the wallet's first ever transaction.

    Uses first_tx_timestamp from fetcher when available (dedicated
    asc-sorted API call). Falls back to min(timestamps) of fetched
    batch only if the dedicated call failed.
    """
    if first_tx_timestamp and first_tx_timestamp > 0:
        first_seen = first_tx_timestamp
    else:
        all_txns   = eth_txns + token_txns
        timestamps = [tx["timestamp"] for tx in all_txns if tx["timestamp"] > 0]
        if not timestamps:
            return {
                "value":         0,
                "display":       "Unknown",
                "first_seen_ts": None,
            }
        first_seen = min(timestamps)

    age_days = (time.time() - first_seen) / 86400

    return {
        "value":         round(age_days, 1),
        "display":       format_age(age_days),   # "10.5 years" or "22 days"
        "first_seen_ts": first_seen,
    }


def _compute_round_transactions(eth_txns: list) -> dict:
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
                break

    return {
        "value":      len(round_txns),
        "round_txns": round_txns[:10],
    }


def _compute_incoming_only(eth_txns: list, token_txns: list) -> dict:
    all_txns     = eth_txns + token_txns
    has_outgoing = any(tx.get("direction") == "out" for tx in all_txns)
    has_incoming = any(tx.get("direction") == "in"  for tx in all_txns)

    return {
        "value":        not has_outgoing and has_incoming,
        "has_outgoing": has_outgoing,
        "has_incoming": has_incoming,
    }


def _compute_token_diversity(token_txns: list) -> dict:
    symbols = [tx.get("token_symbol", "UNKNOWN") for tx in token_txns]
    counts  = Counter(symbols)

    return {
        "value":           len(counts),
        "token_breakdown": dict(counts.most_common(10)),
    }


# ── scorer ────────────────────────────────────────────────────────────────────

def _compute_score(signals: dict) -> int:
    score = 0

    mixer = signals["mixer_exposure"]["value"]
    if mixer >= 3:
        score += 35
    elif mixer >= 1:
        score += 20

    fan_out = signals["fan_out_ratio"]["value"]
    if fan_out >= FAN_OUT_HIGH:
        score += 20
    elif fan_out >= FAN_OUT_MEDIUM:
        score += 10

    velocity = signals["transaction_velocity"]["value"]
    if velocity >= VELOCITY_HIGH:
        score += 15
    elif velocity >= VELOCITY_MEDIUM:
        score += 7

    round_count = signals["round_transactions"]["value"]
    if round_count >= ROUND_TXN_HIGH:
        score += 10
    elif round_count >= ROUND_TXN_MEDIUM:
        score += 5

    age = signals["account_age_days"]["value"]
    if 0 < age <= ACCOUNT_AGE_NEW:
        score += 10
    elif age <= ACCOUNT_AGE_MEDIUM:
        score += 5

    diversity = signals["token_diversity"]["value"]
    if diversity >= TOKEN_DIVERSITY_HIGH:
        score += 5
    elif diversity >= TOKEN_DIVERSITY_MED:
        score += 2

    if signals["incoming_only"]["value"]:
        score += 5

    return min(score, 100)


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
    """
    eth_txns           = wallet_data.get("eth_transactions", [])
    token_txns         = wallet_data.get("token_transfers",  [])
    address            = wallet_data.get("address", "unknown")
    # FIX 9: read first_tx_timestamp from wallet_data and pass it through
    first_tx_timestamp = wallet_data.get("first_tx_timestamp", 0)

    signals = {
        "transaction_velocity": _compute_velocity(eth_txns, token_txns),
        "fan_out_ratio":        _compute_fan_out(eth_txns),
        "mixer_exposure":       _compute_mixer_exposure(eth_txns, token_txns),
        "account_age_days":     _compute_account_age(
                                    eth_txns,
                                    token_txns,
                                    first_tx_timestamp=first_tx_timestamp,
                                ),
        "round_transactions":   _compute_round_transactions(eth_txns),
        "incoming_only":        _compute_incoming_only(eth_txns, token_txns),
        "token_diversity":      _compute_token_diversity(token_txns),
    }

    score = _compute_score(signals)
    label = _risk_label(score)
    flags = _build_flags(signals, score)

    return {
        "address":    address,
        "risk_score": score,
        "risk_label": label,
        "signals":    signals,
        "top_flags":  flags,
        "metadata": {
            "eth_balance":       wallet_data.get("balance_eth", 0.0),
            "total_tx_analysed": wallet_data.get("total_tx_count", 0),
        },
    }


def _build_flags(signals: dict, score: int) -> list:
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
        # FIX 10: use age["display"] not age["value"] — shows "22 days"
        # not "22.0", and shows "1.2 years" not "438.0" for older wallets
        flags.append(
            f"Young wallet: first seen {age['display']} ago"
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
# python src/risk_engine.py

if __name__ == "__main__":

    print("\nRisk engine — synthetic data self-test")
    print("-" * 60)

    now = int(time.time())

    synthetic_eth_txns = []
    for i in range(80):
        synthetic_eth_txns.append({
            "hash":         f"0xabc{i:04x}",
            "timestamp":    now - (i * 300),
            "from_address": "0xtargetwalletaddress",
            "to_address":   f"0xunique_recipient_{i:04x}",
            "value_eth":    1.0 if i % 10 == 0 else 0.037,
            "direction":    "out",
            "block_number": 19000000 - i,
            "is_error":     False,
            "gas_used":     21000,
        })

    synthetic_eth_txns.append({
        "hash":         "0xmixerhash001",
        "timestamp":    now - 86400,
        "from_address": "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",
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
        for i in range(20)
    ]

    # Simulate 3-day-old wallet via first_tx_timestamp
    first_tx_timestamp = now - (3 * 86400)

    synthetic_wallet = {
        "address":            "0xtargetwalletaddress",
        "balance_eth":        4.2,
        "first_tx_timestamp": first_tx_timestamp,
        "eth_transactions":   synthetic_eth_txns,
        "token_transfers":    synthetic_token_txns,
        "fetch_errors":       [],
        "total_tx_count":     len(synthetic_eth_txns) + len(synthetic_token_txns),
    }

    result = extract_risk_signals(synthetic_wallet)

    print(f"  Risk score:    {result['risk_score']} / 100")
    print(f"  Risk label:    {result['risk_label']}")

    print("\n  Signals:")
    for name, sig in result["signals"].items():
        display = sig.get("display", sig["value"])
        print(f"    {name:<28} {display}")

    print("\n  Flags raised:")
    for flag in result["top_flags"]:
        print(f"    - {flag}")

    # Age display tests
    print("\n  Age display tests:")
    print(f"    10.5 years:  {format_age(10.5 * 365.25)}")
    print(f"    22 days:     {format_age(22)}")
    print(f"    364 days:    {format_age(364)}")
    print(f"    365 days:    {format_age(365)}")

    ok = result["risk_label"] == "HIGH"
    print(f"\n  [{'OK' if ok else 'FAIL'}] Expected HIGH — got {result['risk_label']}\n")