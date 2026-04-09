import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from websocket import WebSocketApp

REST_BASE = "https://api.derivws.com"


# ------------------------------
# Helpers
# ------------------------------
def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def stddev(values: List[float]) -> float:
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def bollinger(prices: List[float], length: int, mult: float) -> Optional[Dict[str, float]]:
    if len(prices) < length:
        return None
    sample = prices[-length:]
    basis = mean(sample)
    dev = stddev(sample) * mult
    return {
        "middle": basis,
        "upper": basis + dev,
        "lower": basis - dev,
    }


def moving_average(prices: List[float], length: int) -> Optional[float]:
    if len(prices) < length:
        return None
    return mean(prices[-length:])


def add_log(message: str, level: str = "info") -> None:
    st.session_state.logs.insert(
        0,
        {
            "time": time.strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        },
    )
    st.session_state.logs = st.session_state.logs[:250]


def normalize_text(value: str) -> str:
    return (value or "").strip()


# ------------------------------
# Bot state
# ------------------------------
@dataclass
class BotState:
    pending_proposal: bool = False
    contract_id: Optional[int] = None
    proposal_id: Optional[str] = None
    entry_tick_counter: int = 0
    waiting_for_fresh_cross: bool = True
    last_trade_time: float = 0.0
    prev_middle: Optional[float] = None
    last_price: Optional[float] = None
    connected: bool = False
    in_trade: bool = False
    status: str = "Disconnected"
    ws_url: Optional[str] = None
    tick_subscription_id: Optional[str] = None
    open_contract_subscription_id: Optional[str] = None
    req_id: int = 1
    pending_map: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    balance: Optional[Dict[str, Any]] = None
    contract_info: Optional[Dict[str, Any]] = None
    symbols: List[Dict[str, Any]] = field(default_factory=list)
    ticks: List[float] = field(default_factory=list)


# ------------------------------
# WebSocket manager
# ------------------------------
class DerivWSManager:
    def __init__(self) -> None:
        self.ws_app: Optional[WebSocketApp] = None
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def connect(self, ws_url: str) -> None:
        self.close()

        def on_open(ws):
            st.session_state.bot.connected = True
            st.session_state.bot.status = "Connected"
            add_log("Connected to Deriv WebSocket.", "success")
            self.send({"balance": 1, "subscribe": 1}, {"type": "balance"})
            self.send({"active_symbols": "brief"}, {"type": "active_symbols"})
            self.send(
                {"ticks": st.session_state.form["symbol"], "subscribe": 1},
                {"type": "ticks"},
            )

        def on_message(ws, message: str):
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                add_log("Received invalid JSON from WebSocket.", "error")
                return

            if data.get("error"):
                add_log(f"{data['error'].get('code')}: {data['error'].get('message')}", "error")
                return

            req_id = data.get("req_id")
            context = None
            if req_id:
                context = st.session_state.bot.pending_map.pop(req_id, None)

            msg_type = data.get("msg_type")
            if msg_type == "balance":
                st.session_state.bot.balance = data.get("balance")
            elif msg_type == "active_symbols":
                st.session_state.bot.symbols = data.get("active_symbols", [])
            elif msg_type == "tick":
                tick = data.get("tick") or {}
                quote = tick.get("quote")
                if quote is not None:
                    st.session_state.bot.ticks.append(float(quote))
                    st.session_state.bot.ticks = st.session_state.bot.ticks[-500:]
                    st.session_state.last_price = float(quote)
                    self.evaluate_strategy()
                if data.get("subscription", {}).get("id") and not st.session_state.bot.tick_subscription_id:
                    st.session_state.bot.tick_subscription_id = data["subscription"]["id"]
            elif msg_type == "proposal":
                self.handle_proposal(data, context)
            elif msg_type == "buy":
                self.handle_buy(data, context)
            elif msg_type == "proposal_open_contract":
                self.handle_open_contract(data.get("proposal_open_contract"), data.get("subscription", {}).get("id"))
            elif msg_type == "sell":
                add_log("Sell response received.", "success")

        def on_error(ws, error):
            add_log(f"WebSocket error: {error}", "error")

        def on_close(ws, code, msg):
            st.session_state.bot.connected = False
            st.session_state.bot.in_trade = False
            st.session_state.bot.status = "Disconnected"
            add_log("WebSocket disconnected.", "warn")

        self.ws_app = WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        self.thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        with self.lock:
            if self.ws_app:
                try:
                    self.ws_app.close()
                except Exception:
                    pass
            self.ws_app = None
            self.thread = None

    def send(self, payload: dict, context: Optional[dict] = None) -> None:
        with self.lock:
            if not self.ws_app:
                return
            req_id = st.session_state.bot.req_id
            st.session_state.bot.req_id += 1
            payload = dict(payload)
            payload["req_id"] = req_id
            if context:
                st.session_state.bot.pending_map[req_id] = context
            try:
                self.ws_app.send(json.dumps(payload))
            except Exception as exc:
                add_log(f"Send failed: {exc}", "error")

    def evaluate_strategy(self) -> None:
        if not st.session_state.auto_trading or not st.session_state.bot.connected:
            st.session_state.bot.last_price = st.session_state.last_price
            return

        prices = st.session_state.bot.ticks
        form = st.session_state.form
        bb = bollinger(prices, int(form["bb_length"]), float(form["bb_std"]))
        trend = moving_average(prices, int(form["trend_length"]))
        current_price = st.session_state.last_price
        state = st.session_state.bot

        if not bb or trend is None or current_price is None:
            state.last_price = current_price
            return

        middle_rising = state.prev_middle is not None and bb["middle"] > state.prev_middle
        crossed_up = (
            state.last_price is not None
            and state.last_price <= bb["middle"]
            and current_price > bb["middle"]
        )
        cooldown_ok = (time.time() - state.last_trade_time) * 1000 >= int(form["cooldown_ms"])
        trend_ok = current_price > trend

        if not state.in_trade and current_price <= bb["middle"]:
            state.waiting_for_fresh_cross = True

        if state.in_trade:
            state.entry_tick_counter += 1
            weak_exit = current_price < bb["middle"]
            if weak_exit and state.contract_id:
                add_log("Weakness exit triggered.", "warn")
                self.send({"sell": state.contract_id, "price": 0}, {"type": "sell"})
            elif state.entry_tick_counter >= int(form["hold_ticks"]) and state.contract_id:
                add_log(f"Hold limit reached after {form['hold_ticks']} ticks.", "info")
                self.send({"sell": state.contract_id, "price": 0}, {"type": "sell"})

        entry_allowed = (
            not state.in_trade
            and not state.pending_proposal
            and state.waiting_for_fresh_cross
            and crossed_up
            and middle_rising
            and trend_ok
            and cooldown_ok
        )

        if entry_allowed:
            state.pending_proposal = True
            state.waiting_for_fresh_cross = False
            add_log(f"Signal confirmed on {form['symbol']}. Requesting proposal...", "success")
            self.send(
                {
                    "proposal": 1,
                    "amount": float(form["amount"]),
                    "basis": "stake",
                    "contract_type": "ACCU",
                    "currency": form["currency"],
                    "underlying_symbol": form["symbol"],
                    "growth_rate": float(form["growth_rate"]),
                    "subscribe": 0,
                },
                {"type": "proposal"},
            )

        state.prev_middle = bb["middle"]
        state.last_price = current_price

    def handle_proposal(self, data: dict, context: Optional[dict]) -> None:
        if not context or context.get("type") != "proposal":
            return
        proposal = data.get("proposal") or {}
        proposal_id = proposal.get("id")
        ask_price = float(proposal.get("ask_price", 0) or 0)
        if not proposal_id:
            st.session_state.bot.pending_proposal = False
            add_log("Proposal missing ID.", "error")
            return

        st.session_state.bot.proposal_id = proposal_id
        add_log(f"Proposal received at {ask_price}. Buying...", "info")
        self.send({"buy": proposal_id, "price": ask_price, "subscribe": 1}, {"type": "buy"})

    def handle_buy(self, data: dict, context: Optional[dict]) -> None:
        if not context or context.get("type") != "buy":
            return
        buy = data.get("buy") or {}
        contract_id = buy.get("contract_id")
        if not contract_id:
            st.session_state.bot.pending_proposal = False
            add_log("Buy response missing contract ID.", "error")
            return

        state = st.session_state.bot
        state.pending_proposal = False
        state.contract_id = contract_id
        state.entry_tick_counter = 0
        state.in_trade = True
        state.contract_info = buy
        add_log(f"Bought contract {contract_id}.", "success")

        self.send(
            {"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1},
            {"type": "open_contract"},
        )

    def handle_open_contract(self, contract: Optional[dict], subscription_id: Optional[str]) -> None:
        if not contract:
            return
        state = st.session_state.bot
        if subscription_id and not state.open_contract_subscription_id:
            state.open_contract_subscription_id = subscription_id
        state.contract_info = contract

        if contract.get("is_sold"):
            profit = contract.get("profit")
            add_log(
                f"Contract closed. Profit: {profit}",
                "success" if float(profit or 0) >= 0 else "error",
            )
            state.contract_id = None
            state.proposal_id = None
            state.entry_tick_counter = 0
            state.pending_proposal = False
            state.last_trade_time = time.time()
            state.in_trade = False


# ------------------------------
# REST
# ------------------------------
def load_accounts(token: str) -> List[Dict[str, Any]]:
    token = normalize_text(token)
    if not token:
        raise ValueError("API token is required.")

    response = requests.get(
        f"{REST_BASE}/trading/v1/options/accounts",
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("accounts", [])



def get_ws_url(token: str, account_id: str) -> str:
    token = normalize_text(token)
    account_id = normalize_text(account_id)
    if not token:
        raise ValueError("API token is required.")
    if not account_id:
        raise ValueError("Account ID is required.")

    response = requests.post(
        f"{REST_BASE}/trading/v1/options/accounts/{account_id}/otp",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    ws_url = data.get("url") or data.get("websocket_url") or data.get("ws_url")
    if not ws_url:
        raise ValueError("OTP response did not include a WebSocket URL.")
    return ws_url


# ------------------------------
# Streamlit setup
# ------------------------------
st.set_page_config(page_title="Deriv Trading Dashboard", layout="wide")

if "form" not in st.session_state:
    st.session_state.form = {
        "token": "",
        "account_id": "",
        "symbol": "R_10",
        "amount": 1.0,
        "currency": "USD",
        "growth_rate": 0.03,
        "bb_length": 20,
        "bb_std": 2.0,
        "trend_length": 50,
        "hold_ticks": 5,
        "cooldown_ms": 8000,
    }
if "accounts" not in st.session_state:
    st.session_state.accounts = []
if "logs" not in st.session_state:
    st.session_state.logs = []
if "bot" not in st.session_state:
    st.session_state.bot = BotState()
if "last_price" not in st.session_state:
    st.session_state.last_price = None
if "auto_trading" not in st.session_state:
    st.session_state.auto_trading = False
if "ws_manager" not in st.session_state:
    st.session_state.ws_manager = DerivWSManager()

st.title("Deriv Trading Dashboard")
st.caption("Connect your Deriv account, stream live prices, and run your Bollinger-based bot. Use a demo account first.")

left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("1. Connect account")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.form["token"] = normalize_text(
            st.text_input("API Token", value=st.session_state.form["token"], type="password")
        )
    with c2:
        account_mode = st.selectbox("Account mode", ["demo", "real"])
        st.session_state.form["account_id"] = normalize_text(
            st.text_input("Account ID", value=st.session_state.form["account_id"])
        )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Load Accounts", use_container_width=True):
            try:
                st.session_state.accounts = load_accounts(st.session_state.form["token"])
                add_log(f"Loaded {len(st.session_state.accounts)} account(s).", "success")
                preferred = next(
                    (
                        acc for acc in st.session_state.accounts
                        if account_mode in str(acc.get("account_type", "")).lower()
                    ),
                    None,
                )
                if preferred:
                    st.session_state.form["account_id"] = preferred.get("account_id", "")
            except Exception as exc:
                add_log(f"Failed to load accounts: {exc}", "error")
    with col_b:
        if st.button("Connect", use_container_width=True):
            try:
                ws_url = get_ws_url(
                    st.session_state.form["token"],
                    st.session_state.form["account_id"],
                )
                st.session_state.bot.ws_url = ws_url
                st.session_state.bot.status = "Connecting..."
                st.session_state.ws_manager.connect(ws_url)
            except Exception as exc:
                add_log(f"Connection failed: {exc}", "error")
    with col_c:
        if st.button("Disconnect", use_container_width=True):
            st.session_state.auto_trading = False
            st.session_state.ws_manager.close()
            st.session_state.bot.connected = False
            st.session_state.bot.status = "Disconnected"
            add_log("Disconnected by user.", "warn")

    if st.session_state.accounts:
        st.write("Available accounts")
        for acc in st.session_state.accounts:
            cols = st.columns([2, 2, 1, 1])
            cols[0].write(acc.get("account_id", "—"))
            cols[1].write(acc.get("account_type", "—"))
            cols[2].write(acc.get("currency", "—"))
            if cols[3].button("Use", key=f"use-{acc.get('account_id')}"):
                st.session_state.form["account_id"] = acc.get("account_id", "")

    st.subheader("2. Market and bot settings")
    s1, s2, s3 = st.columns(3)
    markets = ["R_10", "R_25", "R_100"]
    current_symbol = st.session_state.form["symbol"] if st.session_state.form["symbol"] in markets else "R_10"
    with s1:
        st.session_state.form["symbol"] = st.selectbox("Market", markets, index=markets.index(current_symbol))
        st.session_state.form["bb_length"] = st.number_input("BB length", min_value=5, value=int(st.session_state.form["bb_length"]))
        st.session_state.form["hold_ticks"] = st.number_input("Hold ticks", min_value=1, value=int(st.session_state.form["hold_ticks"]))
    with s2:
        st.session_state.form["amount"] = st.number_input("Stake", min_value=0.35, value=float(st.session_state.form["amount"]), step=0.1)
        st.session_state.form["bb_std"] = st.number_input("BB std dev", min_value=0.1, value=float(st.session_state.form["bb_std"]), step=0.1)
        st.session_state.form["cooldown_ms"] = st.number_input("Cooldown (ms)", min_value=0, value=int(st.session_state.form["cooldown_ms"]), step=1000)
    with s3:
        st.session_state.form["growth_rate"] = st.number_input("Growth rate", min_value=0.01, value=float(st.session_state.form["growth_rate"]), step=0.01)
        st.session_state.form["trend_length"] = st.number_input("Trend length", min_value=5, value=int(st.session_state.form["trend_length"]))
        st.session_state.form["currency"] = normalize_text(st.text_input("Currency", value=st.session_state.form["currency"])) or "USD"

    a1, a2 = st.columns(2)
    with a1:
        if st.button("Refresh Market Feed", use_container_width=True, disabled=not st.session_state.bot.connected):
            st.session_state.bot.ticks = []
            st.session_state.bot.prev_middle = None
            st.session_state.bot.last_price = None
            st.session_state.bot.waiting_for_fresh_cross = True
            st.session_state.ws_manager.send(
                {"ticks": st.session_state.form["symbol"], "subscribe": 1},
                {"type": "ticks"},
            )
            add_log(f"Subscribed to ticks for {st.session_state.form['symbol']}", "info")
    with a2:
        if st.button("Start / Stop Bot", use_container_width=True, disabled=not st.session_state.bot.connected):
            st.session_state.auto_trading = not st.session_state.auto_trading
            add_log(
                "Auto trading enabled." if st.session_state.auto_trading else "Auto trading disabled.",
                "success" if st.session_state.auto_trading else "warn",
            )

with right:
    st.subheader("Live summary")
    metric1, metric2 = st.columns(2)
    metric1.metric("Status", st.session_state.bot.status)
    metric2.metric("Trade status", "In Trade" if st.session_state.bot.in_trade else "Idle")

    metric3, metric4 = st.columns(2)
    metric3.metric("Last price", st.session_state.last_price if st.session_state.last_price is not None else "—")
    bal = st.session_state.bot.balance or {}
    metric4.metric("Balance", f"{bal.get('currency', '')} {bal.get('balance', '—')}")

    bb = bollinger(st.session_state.bot.ticks, int(st.session_state.form["bb_length"]), float(st.session_state.form["bb_std"]))
    trend = moving_average(st.session_state.bot.ticks, int(st.session_state.form["trend_length"]))
    metric5, metric6 = st.columns(2)
    metric5.metric("BB middle", round(bb["middle"], 4) if bb else "—")
    metric6.metric("Trend MA", round(trend, 4) if trend is not None else "—")

    st.subheader("Open contract")
    if st.session_state.bot.contract_info:
        st.json(st.session_state.bot.contract_info)
    else:
        st.info("No contract yet.")

    st.subheader("Activity log")
    if st.session_state.logs:
        for item in st.session_state.logs:
            st.write(f"[{item['time']}] {item['message']}")
    else:
        st.caption("No events yet.")

st.caption("Run with: pip install streamlit requests websocket-client && streamlit run app.py")
