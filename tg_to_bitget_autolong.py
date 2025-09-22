import asyncio
import os
import time
import json
import hmac
import hashlib
import base64
import re
import math
import websockets
import json
from typing import Optional, Tuple

try:
    import uvloop

    uvloop.install()
except Exception:
    pass

import aiohttp
from dotenv import load_dotenv

load_dotenv()

WS_URL = os.getenv("WS_URL", "")

BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
BITGET_API_SECRET = os.getenv("BITGET_API_SECRET", "")
BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")

MODE = os.getenv("MODE", "mix").lower()
MARGIN_COIN = os.getenv("MARGIN_COIN", "USDT")
SIZE_CONTRACTS = os.getenv("SIZE_CONTRACTS", "")
QUOTE_USDT = float(os.getenv("QUOTE_USDT", "10"))
LEVERAGE = int(os.getenv("LEVERAGE", "5"))

DRY_RUN = os.getenv("DRY_RUN", "1") == "1"
PRINT_LOGS = os.getenv("PRINT_LOGS", "1") == "1"

BASE_URL = os.getenv("BITGET_BASE_URL", "https://api.bitget.com")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "2.0"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "1.0"))
KEEPALIVE_SECONDS = int(os.getenv("KEEPALIVE_SECONDS", "30"))

PAIR_RE = re.compile(
    r"\((?P<base>[A-Z0-9]{2,10})\).*?KRW",
    re.IGNORECASE,
)


def _log(*a):
    if PRINT_LOGS:
        print(*a, flush=True)


def _ts_ms():
    return str(int(time.time() * 1000))


def _sign(message: str, secret_key: str) -> str:
    mac = hmac.new(
        secret_key.encode("utf-8"), message.encode("utf-8"), digestmod=hashlib.sha256
    )
    return base64.b64encode(mac.digest()).decode("utf-8")


def pre_hash(timestamp, method, request_path, body):
    return f"{timestamp}{method.upper()}{request_path}{body}"


def _headers(timestamp: str, method: str, request_path: str, body: str) -> dict:
    signature = _sign(
        pre_hash(timestamp, method, request_path, body), BITGET_API_SECRET
    )
    return {
        "ACCESS-KEY": BITGET_API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": str(timestamp),
        "ACCESS-PASSPHRASE": BITGET_PASSPHRASE,
        "Content-Type": "application/json",
    }


def normalize_symbol_from_text(text: str) -> Optional[Tuple[str, str]]:
    m = PAIR_RE.search(text.upper())
    if not m:
        return None
    base = m.group("base").upper()
    if not base:
        return None
    return base, "USDT"


def floor_to_n_decimals(value: float, n: int) -> float:
    factor = 10**n
    floored = math.floor(float(value) * factor) / factor
    return floored


async def get_ticker(
    session: aiohttp.ClientSession, symbol: str, mix: bool
) -> Optional[float]:
    path = (
        f"/api/v2/mix/market/ticker?productType=USDT-FUTURES&symbol={symbol}"
        if mix
        else f"/api/v2/spot/market/ticker?symbol={symbol}"
    )
    url = BASE_URL + path
    try:
        async with session.get(url, timeout=HTTP_TIMEOUT) as r:
            j = await r.json()
            data = j.get("data") or {}
            last = data[0].get("lastPr")
            return float(last) if last else None
    except Exception as e:
        _log("ticker error:", e)
        return None


async def place_mix_market_buy(
    session: aiohttp.ClientSession, instrument: str, size_contracts: str
) -> dict:
    path = "/api/v2/mix/order/place-order"
    url = BASE_URL + path
    body_dict = {
        "symbol": instrument,
        "marginCoin": MARGIN_COIN,
        "size": size_contracts,
        "side": "buy",
        "tradeSide": "open",
        "orderType": "market",
        "productType": "USDT-FUTURES",
        "marginMode": "crossed",
    }
    body = json.dumps(body_dict, separators=(",", ":"))
    ts = _ts_ms()
    headers = _headers(ts, "POST", path, body)
    async with session.post(url, data=body, headers=headers, timeout=HTTP_TIMEOUT) as r:
        return await r.json()


async def place_spot_market_buy(
    session: aiohttp.ClientSession, symbol: str, quote_amount: float
) -> dict:
    path = "/api/v2/spot/trade/place-order"
    url = BASE_URL + path
    body_dict = {
        "symbol": symbol,
        "side": "buy",
        "orderType": "market",
        "force": "gtc",
        "quoteOrderQty": f"{quote_amount:.8f}",
    }
    body = json.dumps(body_dict, separators=(",", ":"))
    ts = _ts_ms()
    headers = _headers(ts, "POST", path, body)
    async with session.post(url, data=body, headers=headers, timeout=HTTP_TIMEOUT) as r:
        return await r.json()


async def compute_mix_size(session: aiohttp.ClientSession, instrument: str) -> str:
    if SIZE_CONTRACTS:
        return str(SIZE_CONTRACTS)
    last = await get_ticker(session, instrument, mix=True)
    if not last or last <= 0:
        return "1"
    est_qty = max(1.0, (QUOTE_USDT / last))
    est_qty = max(0.001, float(f"{est_qty:.4f}"))
    return str(est_qty)


async def set_leverage(session, instrument: str, leverage: int, hold_side="long"):
    path = "/api/v2/mix/account/set-leverage"
    url = BASE_URL + path
    body_dict = {
        "symbol": instrument,
        "marginCoin": MARGIN_COIN,
        "leverage": str(leverage),
        "holdSide": hold_side,
        "productType": "USDT-FUTURES",
        "marginMode": "crossed",
    }
    body = json.dumps(body_dict, separators=(",", ":"))
    ts = _ts_ms()
    headers = _headers(ts, "POST", path, body)
    async with session.post(url, data=body, headers=headers, timeout=HTTP_TIMEOUT) as r:
        return await r.json()


async def place_take_profit(
    session: aiohttp.ClientSession, instrument: str, size: str, tp_price: float
) -> dict:
    path = "/api/v2/mix/order/place-tpsl-order"
    url = BASE_URL + path
    body_dict = {
        "symbol": instrument,
        "productType": "USDT-FUTURES",
        "marginCoin": MARGIN_COIN,
        "planType": "profit_plan",
        "triggerPrice": str(tp_price),
        "holdSide": "long",
        "size": size,
        "orderType": "market",
        "marginMode": "crossed",
    }
    body = json.dumps(body_dict, separators=(",", ":"))
    ts = _ts_ms()
    headers = _headers(ts, "POST", path, body)
    async with session.post(url, data=body, headers=headers, timeout=HTTP_TIMEOUT) as r:
        return await r.json()


async def handle_signal(session: aiohttp.ClientSession, text: str):
    pair = normalize_symbol_from_text(text)
    if not pair:
        return
    base, quote = pair
    sym = f"{base}{quote}"

    if DRY_RUN:
        _log(f"[DRY_RUN] Parsed {base}/{quote} ➜ MIX:{sym} SPOT:{sym}")
        return

    try:
        if MODE == "mix":
            lev_res = await set_leverage(session, sym, LEVERAGE, hold_side="long")
            _log(f"[MIX] Set leverage {LEVERAGE}x =>", lev_res)

            size = await compute_mix_size(session, sym)
            res = await place_mix_market_buy(session, sym, size)
            _log(f"[MIX] BUY {sym} size={size} =>", res)

            last_price = await get_ticker(session, sym, mix=True)
            if last_price:
                tp_price_first_part = floor_to_n_decimals(last_price * 1.15, 4)
                tp_price_second_part = floor_to_n_decimals(last_price * 1.2, 4)
                tp_first_res = await place_take_profit(
                    session,
                    sym,
                    str(floor_to_n_decimals(float(size) / 2, 1)),
                    tp_price_first_part,
                )
                _log(f"[MIX] TP {tp_price_first_part:.4f} =>", tp_first_res)
                tp_second_res = await place_take_profit(
                    session,
                    sym,
                    str(floor_to_n_decimals(float(size) / 2, 1)),
                    tp_price_second_part,
                )
                _log(f"[MIX] TP {tp_price_second_part:.4f} =>", tp_second_res)
        else:
            res = await place_spot_market_buy(session, sym, QUOTE_USDT)
            _log(f"[SPOT] BUY {sym} quote={QUOTE_USDT} =>", res)
    except Exception as e:
        _log("order error:", e)


async def connect_websocket():
    retry_delay = 1

    timeout = aiohttp.ClientTimeout(
        total=None,
        connect=CONNECT_TIMEOUT,
        sock_read=HTTP_TIMEOUT,
        sock_connect=CONNECT_TIMEOUT,
    )
    connector = aiohttp.TCPConnector(
        limit=10,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
        keepalive_timeout=KEEPALIVE_SECONDS,
    )
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        while True:
            try:
                async with websockets.connect(WS_URL) as websocket:
                    _log("WebSocket connection established")

                    while True:
                        try:
                            message = await websocket.recv()
                            try:
                                message_data = json.loads(message)
                                _log(f"Received message: {message_data}")
                                if "source" in message_data:
                                    if message_data["source"] == "UPBIT":
                                        await handle_signal(
                                            session, message_data["title"]
                                        )
                            except json.JSONDecodeError:
                                _log(f"Received non-JSON message: {message}")
                        except websockets.exceptions.ConnectionClosedError:
                            _log("WebSocket connection closed. Retrying...")
                            break

            except (
                websockets.exceptions.InvalidURI,
                websockets.exceptions.InvalidHandshake,
            ):
                _log(
                    f"Failed to connect to WebSocket server. Retrying in {retry_delay} seconds..."
                )

            except ConnectionRefusedError:
                _log(f"Connection refused. Retrying in {retry_delay} seconds...")

            except Exception as e:
                _log(f"Unexpected error: {e}. Retrying in {retry_delay} seconds...")

            retry_delay = 1

            await asyncio.sleep(retry_delay)


if __name__ == "__main__":
    try:
        asyncio.run(connect_websocket())
    except KeyboardInterrupt:
        pass
