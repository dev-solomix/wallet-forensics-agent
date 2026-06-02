"""
fetcher.py
----------
Pulls raw on-chain data for a given Ethereum wallet address
from the Etherscan API V2.

Endpoints used:
  - account/txlist        → normal ETH transactions (last 100)
  - account/tokentx       → ERC-20 token transfers (last 100)
  - account/balance       → current ETH balance

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
API_KEY = os.getenv("ETHERSCAN_API_KEY")

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

    # Etherscan API-level errors
    if str(data.get("status")) == "0":
        message = data.get("message", "Unknown error")
        result = data.get("result", "")

        # Empty wallet is not fatal
        if "No transactions found" in str(result):
            return {"status": "0", "result": []}

        raise RuntimeError(
            f"Etherscan API error: {message} — {result}"
        )

    return data


# ── core fetchers ─────────────────────────────────────────────────────────────

def fetch_eth_transactions(address: str, limit: int = 100) -> list:
    """
    Fetch recent ETH transactions.
    """

    if not API_KEY:
        raise RuntimeError(
            "ETHERSCAN_API_KEY missing from .env"
        )

    params = {
        "chainid": CHAIN_ID,
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": 1,
        "offset": limit,
        "sort": "desc",
        "apikey": API_KEY,
    }

    time.sleep(REQUEST_DELAY_SECONDS)

    data = _safe_get(ETHERSCAN_BASE_URL, params)
    raw_txns = data.get("result", [])

    cleaned = []

    for tx in raw_txns:
        cleaned.append({
            "hash": tx.get("hash", ""),
            "timestamp": int(tx.get("timeStamp", 0)),
            "from_address": tx.get("from", "").lower(),
            "to_address": tx.get("to", "").lower(),
            "value_eth": _wei_to_eth(tx.get("value", "0")),
            "direction": (
                "out"
                if tx.get("from", "").lower() == address.lower()
                else "in"
            ),
            "block_number": int(tx.get("blockNumber", 0)),
            "is_error": tx.get("isError", "0") == "1",
            "gas_used": int(tx.get("gasUsed", 0)),
        })

    return cleaned


def fetch_token_transfers(address: str, limit: int = 100) -> list:
    """
    Fetch recent ERC-20 token transfers.
    """

    if not API_KEY:
        raise RuntimeError(
            "ETHERSCAN_API_KEY missing from .env"
        )

    params = {
        "chainid": CHAIN_ID,
        "module": "account",
        "action": "tokentx",
        "address": address,
        "page": 1,
        "offset": limit,
        "sort": "desc",
        "apikey": API_KEY,
    }

    time.sleep(REQUEST_DELAY_SECONDS)

    data = _safe_get(ETHERSCAN_BASE_URL, params)
    raw_transfers = data.get("result", [])

    cleaned = []

    for tx in raw_transfers:

        decimals = int(tx.get("tokenDecimal", 18) or 18)

        try:
            value = int(tx.get("value", 0)) / (10 ** decimals)
        except (ValueError, ZeroDivisionError):
            value = 0.0

        cleaned.append({
            "hash": tx.get("hash", ""),
            "timestamp": int(tx.get("timeStamp", 0)),
            "from_address": tx.get("from", "").lower(),
            "to_address": tx.get("to", "").lower(),
            "token_symbol": tx.get("tokenSymbol", "UNKNOWN"),
            "token_name": tx.get("tokenName", "Unknown Token"),
            "value": value,
            "direction": (
                "out"
                if tx.get("from", "").lower() == address.lower()
                else "in"
            ),
            "contract_address": tx.get(
                "contractAddress",
                ""
            ).lower(),
        })

    return cleaned


def fetch_eth_balance(address: str) -> float:
    """
    Fetch ETH balance.
    """

    if not API_KEY:
        raise RuntimeError(
            "ETHERSCAN_API_KEY missing from .env"
        )

    params = {
        "chainid": CHAIN_ID,
        "module": "account",
        "action": "balance",
        "address": address,
        "tag": "latest",
        "apikey": API_KEY,
    }

    time.sleep(REQUEST_DELAY_SECONDS)

    data = _safe_get(ETHERSCAN_BASE_URL, params)

    return _wei_to_eth(data.get("result", "0"))


# ── public API ────────────────────────────────────────────────────────────────

def fetch_wallet_data(address: str) -> dict:
    """
    Master fetch function.
    """

    address = address.strip().lower()

    result = {
        "address": address,
        "balance_eth": 0.0,
        "eth_transactions": [],
        "token_transfers": [],
        "fetch_errors": [],
        "total_tx_count": 0,
    }

    # ETH balance
    try:
        result["balance_eth"] = fetch_eth_balance(address)
    except RuntimeError as exc:
        result["fetch_errors"].append(f"balance: {exc}")

    # ETH transactions
    try:
        result["eth_transactions"] = fetch_eth_transactions(address)
    except RuntimeError as exc:
        result["fetch_errors"].append(f"eth_transactions: {exc}")

    # Token transfers
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

if __name__ == "__main__":

    # Vitalik Buterin public wallet
    TEST_ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

    print(f"\nFetching data for: {TEST_ADDRESS}")
    print("-" * 60)

    wallet = fetch_wallet_data(TEST_ADDRESS)

    print(f"  Balance:            {wallet['balance_eth']:.4f} ETH")
    print(f"  ETH transactions:   {len(wallet['eth_transactions'])}")
    print(f"  Token transfers:    {len(wallet['token_transfers'])}")
    print(f"  Total tx count:     {wallet['total_tx_count']}")

    if wallet["fetch_errors"]:
        print("\n  Non-fatal errors:")
        for err in wallet["fetch_errors"]:
            print(f"    - {err}")

    if wallet["eth_transactions"]:
        latest = wallet["eth_transactions"][0]

        print("\n  Most recent ETH tx:")
        print(f"    Hash:      {latest['hash']}")
        print(f"    Direction: {latest['direction']}")
        print(f"    Value:     {latest['value_eth']:.6f} ETH")

    if wallet["token_transfers"]:
        latest_token = wallet["token_transfers"][0]

        print("\n  Most recent token transfer:")
        print(f"    Token:     {latest_token['token_symbol']}")
        print(f"    Direction: {latest_token['direction']}")
        print(f"    Value:     {latest_token['value']:.4f}")

    print("\n[OK] Fetcher working correctly.\n")