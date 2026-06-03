"""
fetcher.py
----------
Pulls raw on-chain data for a given Ethereum wallet address
from the Etherscan API V2.

Endpoints used:
  - account/txlist (asc, limit 1) → true first transaction timestamp
  - account/txlist (desc)         → normal ETH transactions (last 100)
  - account/tokentx               → ERC-20 token transfers (last 100)
  - account/balance               → current ETH balance

Requires:
  - ETHERSCAN_API_KEY in .env
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ── constants ────────────────────────────────────────────────────────────────

ETHERSCAN_BASE_URL = "https://api.etherscan.io/v2/api"

# FIX 3: API_KEY must be defined BEFORE the Streamlit fallback block.
# In the old code the fallback referenced API_KEY before it was assigned,
# causing NameError on every cold start.
API_KEY = os.getenv("ETHERSCAN_API_KEY")

# FIX 4: Streamlit secrets fallback comes AFTER os.getenv, not before.
if not API_KEY:
    try:
        import streamlit as st
        API_KEY = st.secrets.get("ETHERSCAN_API_KEY", "")
    except Exception:
        pass

# Ethereum Mainnet
CHAIN_ID = 1

# Etherscan free tier: 5 calls/sec
REQUEST_DELAY_SECONDS = 0.25


# ── helpers ──────────────────────────────────────────────────────────────────

def _wei_to_eth(wei_str: str) -> float:
    """Convert Wei string to ETH float."""
    try:
        return int(wei_str) / 1e18
    except (ValueError, TypeError):
        return 0.0


def _safe_get(url: str, params: dict) -> dict:
    """
    Safe wrapper around requests.get with clean error handling.
    """
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError("Etherscan request timed out.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not connect to Etherscan.")
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Etherscan returned HTTP {exc.response.status_code}"
        )

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError("Invalid JSON response from Etherscan.")

    if str(data.get("status")) == "0":
        message = data.get("message", "Unknown error")
        result  = data.get("result", "")
        if "No transactions found" in str(result):
            return {"status": "0", "result": []}
        raise RuntimeError(
            f"Etherscan API error: {message} — {result}"
        )

    return data


# ── core fetchers ─────────────────────────────────────────────────────────────

# FIX 5: fetch_first_transaction_timestamp was missing entirely.
# Without it, account age is derived from the oldest tx in the last 100,
# which gives wrong results for old high-volume wallets (e.g. 10-year
# wallet shows 22 days because that's how far back 100 txns reach).
def fetch_first_transaction_timestamp(address: str) -> int:
    """
    Fetches the wallet's very first transaction by sorting ascending
    with a limit of 1. Returns a Unix timestamp, or 0 if none found.
    """
    if not API_KEY:
        raise RuntimeError("ETHERSCAN_API_KEY missing from .env")

    params = {
        "chainid":    CHAIN_ID,
        "module":     "account",
        "action":     "txlist",
        "address":    address,
        "startblock": 0,
        "endblock":   99999999,
        "page":       1,
        "offset":     1,        # only need the single oldest tx
        "sort":       "asc",    # oldest first
        "apikey":     API_KEY,
    }

    time.sleep(REQUEST_DELAY_SECONDS)
    data = _safe_get(ETHERSCAN_BASE_URL, params)
    txns = data.get("result", [])

    if not txns:
        return 0
    return int(txns[0].get("timeStamp", 0))


def fetch_eth_transactions(address: str, limit: int = 100) -> list:
    """
    Fetch recent ETH transactions (newest first).
    """
    if not API_KEY:
        raise RuntimeError("ETHERSCAN_API_KEY missing from .env")

    params = {
        "chainid":    CHAIN_ID,
        "module":     "account",
        "action":     "txlist",
        "address":    address,
        "startblock": 0,
        "endblock":   99999999,
        "page":       1,
        "offset":     limit,
        "sort":       "desc",
        "apikey":     API_KEY,
    }

    time.sleep(REQUEST_DELAY_SECONDS)
    data     = _safe_get(ETHERSCAN_BASE_URL, params)
    raw_txns = data.get("result", [])

    cleaned = []
    for tx in raw_txns:
        cleaned.append({
            "hash":         tx.get("hash", ""),
            "timestamp":    int(tx.get("timeStamp", 0)),
            "from_address": tx.get("from", "").lower(),
            "to_address":   tx.get("to",   "").lower(),
            "value_eth":    _wei_to_eth(tx.get("value", "0")),
            "direction":    (
                "out"
                if tx.get("from", "").lower() == address.lower()
                else "in"
            ),
            "block_number": int(tx.get("blockNumber", 0)),
            "is_error":     tx.get("isError", "0") == "1",
            "gas_used":     int(tx.get("gasUsed", 0)),
        })

    return cleaned


def fetch_token_transfers(address: str, limit: int = 100) -> list:
    """
    Fetch recent ERC-20 token transfers (newest first).
    """
    if not API_KEY:
        raise RuntimeError("ETHERSCAN_API_KEY missing from .env")

    params = {
        "chainid":  CHAIN_ID,
        "module":   "account",
        "action":   "tokentx",
        "address":  address,
        "page":     1,
        "offset":   limit,
        "sort":     "desc",
        "apikey":   API_KEY,
    }

    time.sleep(REQUEST_DELAY_SECONDS)
    data          = _safe_get(ETHERSCAN_BASE_URL, params)
    raw_transfers = data.get("result", [])

    cleaned = []
    for tx in raw_transfers:
        decimals = int(tx.get("tokenDecimal", 18) or 18)
        try:
            value = int(tx.get("value", 0)) / (10 ** decimals)
        except (ValueError, ZeroDivisionError):
            value = 0.0

        cleaned.append({
            "hash":             tx.get("hash", ""),
            "timestamp":        int(tx.get("timeStamp", 0)),
            "from_address":     tx.get("from", "").lower(),
            "to_address":       tx.get("to",   "").lower(),
            "token_symbol":     tx.get("tokenSymbol", "UNKNOWN"),
            "token_name":       tx.get("tokenName",   "Unknown Token"),
            "value":            value,
            "direction":        (
                "out"
                if tx.get("from", "").lower() == address.lower()
                else "in"
            ),
            "contract_address": tx.get("contractAddress", "").lower(),
        })

    return cleaned


def fetch_eth_balance(address: str) -> float:
    """
    Fetch current ETH balance. Returns float in ETH.
    """
    if not API_KEY:
        raise RuntimeError("ETHERSCAN_API_KEY missing from .env")

    params = {
        "chainid": CHAIN_ID,
        "module":  "account",
        "action":  "balance",
        "address": address,
        "tag":     "latest",
        "apikey":  API_KEY,
    }

    time.sleep(REQUEST_DELAY_SECONDS)
    data = _safe_get(ETHERSCAN_BASE_URL, params)
    return _wei_to_eth(data.get("result", "0"))


# ── public API ────────────────────────────────────────────────────────────────

def fetch_wallet_data(address: str) -> dict:
    """
    Master fetch function. Calls all four endpoints and returns a
    single clean dict. This is the only function agent.py imports.

    Returns:
        {
          "address":            str,
          "balance_eth":        float,
          "first_tx_timestamp": int,    <- FIX 6: true wallet age anchor
          "eth_transactions":   list,
          "token_transfers":    list,
          "fetch_errors":       list,
          "total_tx_count":     int,
        }
    """
    address = address.strip().lower()

    result = {
        "address":            address,
        "balance_eth":        0.0,
        "first_tx_timestamp": 0,        # FIX 6: was missing from return dict
        "eth_transactions":   [],
        "token_transfers":    [],
        "fetch_errors":       [],
        "total_tx_count":     0,
    }

    # 1. True first transaction timestamp (asc sort, limit 1)
    try:
        result["first_tx_timestamp"] = fetch_first_transaction_timestamp(address)
    except RuntimeError as exc:
        result["fetch_errors"].append(f"first_tx: {exc}")

    # 2. ETH balance
    try:
        result["balance_eth"] = fetch_eth_balance(address)
    except RuntimeError as exc:
        result["fetch_errors"].append(f"balance: {exc}")

    # 3. Recent ETH transactions (desc, last 100)
    try:
        result["eth_transactions"] = fetch_eth_transactions(address)
    except RuntimeError as exc:
        result["fetch_errors"].append(f"eth_transactions: {exc}")

    # 4. ERC-20 token transfers (desc, last 100)
    try:
        result["token_transfers"] = fetch_token_transfers(address)
    except RuntimeError as exc:
        result["fetch_errors"].append(f"token_transfers: {exc}")

    result["total_tx_count"] = (
        len(result["eth_transactions"])
        + len(result["token_transfers"])
    )

    return result


# ── quick test ────────────────────────────────────────────────────────────────
# python src/fetcher.py

if __name__ == "__main__":
    import datetime

    TEST_ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

    print(f"\nFetching data for: {TEST_ADDRESS}")
    print("-" * 60)

    wallet = fetch_wallet_data(TEST_ADDRESS)

    print(f"  Balance:            {wallet['balance_eth']:.4f} ETH")
    print(f"  ETH transactions:   {len(wallet['eth_transactions'])}")
    print(f"  Token transfers:    {len(wallet['token_transfers'])}")
    print(f"  Total tx count:     {wallet['total_tx_count']}")

    if wallet["first_tx_timestamp"]:
        first_dt  = datetime.datetime.fromtimestamp(wallet["first_tx_timestamp"])
        age_days  = (datetime.datetime.now() - first_dt).days
        age_years = age_days / 365.25
        print(f"\n  First tx date:  {first_dt.strftime('%Y-%m-%d')}")
        if age_years >= 1:
            print(f"  Wallet age:     {age_years:.1f} years")
        else:
            print(f"  Wallet age:     {age_days} days")

    if wallet["fetch_errors"]:
        print("\n  Non-fatal errors:")
        for err in wallet["fetch_errors"]:
            print(f"    - {err}")

    print("\n[OK] Fetcher working correctly.\n")