#!/usr/bin/env python3
"""Fixed constants, data loading, backtest engine and evaluation for the
Polymarket strategy-research domain.

READ-ONLY for the research agent: the only file an experiment may modify is
strategy.py. The ground-truth metric is val_sharpe printed by evaluate().

Objective
---------
val_sharpe: annualized Sharpe ratio (mean/std * sqrt(365)) of daily
mark-to-market PnL on a fixed $10,000 bankroll, over the VALIDATION window
(markets expiring before VAL_SPLIT). Markets expiring on/after VAL_SPLIT are
the HOLDOUT set: never evaluated during the research loop (see program.md).

Hard constraints (any violation -> val_sharpe is reported as 0):
  max drawdown <= MAX_DD, fills >= MIN_TRADES, active days >= MIN_ACTIVE_DAYS,
  wall-clock <= TIME_BUDGET seconds.

Execution model (deliberately conservative, taker-only)
-------------------------------------------------------
- The strategy is stepped every STEP_MINUTES on a global clock and sees only
  information available at that time (last printed market price, Binance
  minute closes up to now, its own inventory/cash).
- An order can only fill at the market's NEXT recorded print after submission
  (within FILL_WINDOW_MINUTES, else it expires): you transact when the market
  actually traded, at that price, worsened by SLIPPAGE_TICKS.
- Buying NO at YES-print p costs (1-p) plus slippage. Taker fee (when the
  market has fees enabled): shares * FEE_RATE * px * (1 - px), the live
  Polymarket crypto fee formula.
- Per market and per step, at most MAX_FILL_USD notional fills (thin books).
- Positions held at expiry settle at the resolved outcome (0/1).

Strategy API (implement in strategy.py)
---------------------------------------
class Strategy:
    def on_step(self, obs) -> list[tuple]:
        # each order: (engine_index, side, usd, limit_or_None)
        # side in {"BUY_YES","BUY_NO","SELL_YES","SELL_NO"}; limit bounds the
        # acceptable execution price of the leg being traded.

obs fields (numpy arrays over currently active markets unless noted):
    t            float, unix seconds of this step
    idx          int array, stable engine market indices
    kind         int8 (0 = "above $K at noon ET", 1 = "up-or-down vs prev noon")
    asset        int8 (0 = BTC, 1 = ETH)
    strike       float (above: $K; up-or-down: prev-noon close once known, else nan)
    expiry       float unix seconds          tte_min   minutes to expiry
    price        last printed YES price (nan if no print yet)
    age_min      minutes since that print    fee       bool, fees enabled
    tick         minimum price increment
    pos_yes/pos_no  current share inventory  cash      float, free USD
    spot(asset)                 -> latest Binance close
    closes(asset, lookback_min) -> minute closes up to t (numpy array)
"""
import json, os, sys, time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

# ----------------------------- fixed constants -----------------------------
CACHE = os.path.expanduser('~/.cache/autopoly')
START_CAPITAL = 10_000.0
STEP_MINUTES = 5
MAX_FILL_USD = 100.0          # per market per step
SLIPPAGE_TICKS = 1
FEE_RATE = 0.07               # Polymarket crypto taker fee: sh * rate * p(1-p)
FILL_WINDOW_MINUTES = 30
VAL_SPLIT = datetime(2026, 5, 15, tzinfo=timezone.utc).timestamp()
TIME_BUDGET = 300             # seconds, hard wall-clock cap for evaluate()
MAX_DD = 0.25
MIN_TRADES = 30
MIN_ACTIVE_DAYS = 15
NY = ZoneInfo('America/New_York')
SYMS = ['BTCUSDT', 'ETHUSDT']


# --------------------------------- data ------------------------------------
class Data:
    def __init__(self):
        cat = json.load(open(os.path.join(CACHE, 'markets.json')))
        pz = np.load(os.path.join(CACHE, 'prices.npz'), allow_pickle=False)
        ids = {c: i for i, c in enumerate(pz['ids'].tolist())}
        keep = [m for m in cat if m['resolved'] and m['cid'] in ids]
        self.markets = keep
        n = len(keep)
        self.kind = np.array([m['kind'] for m in keep], dtype=np.int8)
        self.asset = np.array([m['asset'] for m in keep], dtype=np.int8)
        self.strike = np.array([m['strike'] for m in keep], dtype=np.float64)
        self.expiry = np.array([m['expiry'] for m in keep], dtype=np.float64)
        self.created = np.array([m['created'] for m in keep], dtype=np.float64)
        self.fee = np.array([m['fee'] for m in keep], dtype=bool)
        self.tick = np.array([m['tick'] for m in keep], dtype=np.float64)
        self.outcome = np.array([m['outcome_yes'] for m in keep], dtype=np.int8)
        self.series = []
        off, ts, ps = pz['offsets'], pz['ts'], pz['ps'].astype(np.float64)
        for m in keep:
            j = ids[m['cid']]
            t_a, p_a = ts[off[j]:off[j + 1]], ps[off[j]:off[j + 1]]
            # strip the leading run of identical prices (pre-trading chart
            # placeholder, typically 0.5): keep only from its last point on
            k = 0
            while k + 1 < len(p_a) and p_a[k + 1] == p_a[0]:
                k += 1
            self.series.append((t_a[k:], p_a[k:]))
        self.binance = []
        for sym in SYMS:
            k = np.load(os.path.join(CACHE, f'binance_{sym}.npz'))['k']
            self.binance.append(k)
        # up-or-down strike (prev-noon close) and the time it becomes known
        self.ud_strike = np.full(n, np.nan)
        self.ud_known = np.full(n, np.inf)
        for i, m in enumerate(keep):
            if m['kind'] != 1:
                continue
            d = date.fromisoformat(m['date'])
            ts_prev = datetime(d.year, d.month, d.day, 12, 1,
                               tzinfo=NY).timestamp() - 86400
            k = self.binance[m['asset']]
            j = int((ts_prev - 60 - k[0, 0]) // 60)
            if 0 <= j < len(k):
                self.ud_strike[i] = k[j, 4]
                self.ud_known[i] = ts_prev

    def spot_idx(self, ai, t):
        k = self.binance[ai]
        return min(int((t - k[0, 0]) // 60), len(k) - 1)


_DATA = None
def load_data():
    global _DATA
    if _DATA is None:
        _DATA = Data()
    return _DATA


# -------------------------------- engine -----------------------------------
class Obs:
    __slots__ = ('t', 'idx', 'kind', 'asset', 'strike', 'expiry', 'tte_min',
                 'price', 'age_min', 'fee', 'tick', 'pos_yes', 'pos_no',
                 'cash', '_data')

    def spot(self, ai):
        k = self._data.binance[ai]
        return k[self._data.spot_idx(ai, self.t), 4]

    def closes(self, ai, lookback_min):
        k = self._data.binance[ai]
        j = self._data.spot_idx(ai, self.t)
        return k[max(0, j - int(lookback_min)):j + 1, 4]


def run_backtest(strategy_factory, split='val', deadline=None):
    d = load_data()
    if split == 'val':
        sel = np.where(d.expiry < VAL_SPLIT)[0]
    else:
        sel = np.where(d.expiry >= VAL_SPLIT)[0]
    if len(sel) == 0:
        raise RuntimeError(f'no markets in split={split}')
    strat = strategy_factory()
    t0 = float(min(d.created[sel].min(), min(d.series[i][0][0] for i in sel)))
    t1 = float(d.expiry[sel].max())
    t0 = np.floor(t0 / 300) * 300
    step = STEP_MINUTES * 60

    cash = START_CAPITAL
    pos = np.zeros((len(d.markets), 2))          # shares: [:,0] YES, [:,1] NO
    ptr = {int(i): 0 for i in sel}               # next unseen print per market
    last_p = np.full(len(d.markets), np.nan)
    last_t = np.full(len(d.markets), np.nan)
    pending = []                                 # (i, side, usd, limit, t_sub)
    n_fills = n_rejects = 0
    fill_days = set()
    equity_curve = []                            # (day_ts, equity)
    live = set(int(i) for i in sel)
    settled = set()
    cur_day = None

    t = t0
    while t <= t1:
        if deadline and time.time() > deadline:
            return None
        # advance prints; collect fills against pending orders
        new_pending = []
        for od in pending:
            i, side, usd, limit, t_sub = od
            ts_a, ps_a = d.series[i]
            p = ptr[i]
            filled = False
            while p < len(ts_a) and ts_a[p] <= t:
                if ts_a[p] > t_sub:              # first print after submission
                    px_yes = float(ps_a[p])
                    tk = d.tick[i] * SLIPPAGE_TICKS
                    if side == 'BUY_YES':
                        px = min(px_yes + tk, 1 - d.tick[i])
                    elif side == 'SELL_YES':
                        px = max(px_yes - tk, d.tick[i])
                    elif side == 'BUY_NO':
                        px = min(1 - px_yes + tk, 1 - d.tick[i])
                    else:
                        px = max(1 - px_yes - tk, d.tick[i])
                    if limit is not None and ((side.startswith('BUY') and px > limit)
                                              or (side.startswith('SELL') and px < limit)):
                        filled = True            # crossed a worse print: cancel
                        break
                    usd_c = min(usd, MAX_FILL_USD)
                    sh = usd_c / px
                    col = 0 if side.endswith('YES') else 1
                    fee = sh * FEE_RATE * px * (1 - px) if d.fee[i] else 0.0
                    if side.startswith('BUY'):
                        if cash < usd_c + fee:
                            n_rejects += 1
                        else:
                            cash -= usd_c + fee
                            pos[i, col] += sh
                            n_fills += 1
                            fill_days.add(int(ts_a[p] // 86400))
                    else:
                        sh = min(sh, pos[i, col])
                        if sh <= 0:
                            n_rejects += 1
                        else:
                            cash += sh * px - sh * FEE_RATE * px * (1 - px) * d.fee[i]
                            pos[i, col] -= sh
                            n_fills += 1
                            fill_days.add(int(ts_a[p] // 86400))
                    filled = True
                    break
                p += 1
            if not filled and t - t_sub < FILL_WINDOW_MINUTES * 60:
                new_pending.append(od)
        pending = new_pending
        # advance pointers / last prices for all live markets
        for i in list(live):
            ts_a, ps_a = d.series[i]
            p = ptr[i]
            while p < len(ts_a) and ts_a[p] <= t:
                p += 1
            ptr[i] = p
            if p > 0:
                last_p[i] = ps_a[p - 1]
                last_t[i] = ts_a[p - 1]
        # settle expiries
        for i in list(live):
            if d.expiry[i] <= t:
                o = float(d.outcome[i])
                cash += pos[i, 0] * o + pos[i, 1] * (1 - o)
                pos[i] = 0
                live.discard(i)
                settled.add(i)
        # daily mark
        day = int(t // 86400)
        if day != cur_day:
            cur_day = day
            mtm = 0.0
            for i in live:
                if not np.isnan(last_p[i]):
                    mtm += pos[i, 0] * last_p[i] + pos[i, 1] * (1 - last_p[i])
            equity_curve.append((t, cash + mtm))
        # build obs over active (created, not expired, has a print)
        act = [i for i in live if d.created[i] <= t and not np.isnan(last_p[i])]
        if act:
            o = Obs()
            o._data = d
            o.t = t
            o.idx = np.array(act)
            o.kind = d.kind[o.idx]
            o.asset = d.asset[o.idx]
            strike = d.strike[o.idx].copy()
            ud = d.kind[o.idx] == 1
            ud_ok = d.ud_known[o.idx] <= t
            strike[ud] = np.where(ud_ok[ud], d.ud_strike[o.idx][ud], np.nan)
            o.strike = strike
            o.expiry = d.expiry[o.idx]
            o.tte_min = (o.expiry - t) / 60
            o.price = last_p[o.idx]
            o.age_min = (t - last_t[o.idx]) / 60
            o.fee = d.fee[o.idx]
            o.tick = d.tick[o.idx]
            o.pos_yes = pos[o.idx, 0]
            o.pos_no = pos[o.idx, 1]
            o.cash = cash
            try:
                orders = strat.on_step(o) or []
            except Exception:
                raise
            for od in orders[:200]:
                i, side, usd, limit = int(od[0]), od[1], float(od[2]), od[3]
                if i in live and usd > 0 and side in ('BUY_YES', 'BUY_NO', 'SELL_YES', 'SELL_NO'):
                    pending.append((i, side, usd, limit, t))
        t += step

    eq = np.array([e for _, e in equity_curve])
    diffs = np.diff(eq)
    sd = diffs.std(ddof=1) if len(diffs) > 2 else 0.0
    sharpe = float(diffs.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    max_dd = float(((peak - eq) / peak).max()) if len(eq) else 1.0
    return dict(sharpe=sharpe, pnl=float(eq[-1] - START_CAPITAL) if len(eq) else 0.0,
                max_dd=max_dd, n_trades=n_fills, n_rejects=n_rejects,
                active_days=len(fill_days), n_markets=len(sel),
                win_days=float((diffs > 0).mean()) if len(diffs) else 0.0)


# ------------------------------- evaluation --------------------------------
def evaluate(strategy_factory):
    split = 'holdout' if os.environ.get('EVAL_HOLDOUT') == '1' else 'val'
    start = time.time()
    try:
        r = run_backtest(strategy_factory, split=split, deadline=start + TIME_BUDGET)
    except Exception as e:
        print(f'CRASH during backtest: {type(e).__name__}: {e}')
        raise
    runtime = time.time() - start
    if r is None:
        print('---')
        print('val_sharpe:       0.000000')
        print('constraint_violated: time_budget_exceeded')
        return
    score = r['sharpe']
    reason = ''
    if r['max_dd'] > MAX_DD:
        score, reason = 0.0, f"max_dd {r['max_dd']:.3f} > {MAX_DD}"
    elif r['n_trades'] < MIN_TRADES:
        score, reason = 0.0, f"n_trades {r['n_trades']} < {MIN_TRADES}"
    elif r['active_days'] < MIN_ACTIVE_DAYS:
        score, reason = 0.0, f"active_days {r['active_days']} < {MIN_ACTIVE_DAYS}"
    tag = 'holdout' if split == 'holdout' else 'val'
    print('---')
    print(f'{tag}_sharpe:       {score:.6f}')
    print(f'{tag}_raw_sharpe:   {r["sharpe"]:.6f}')
    print(f'{tag}_pnl_usd:      {r["pnl"]:.2f}')
    print(f'{tag}_max_dd:       {r["max_dd"]:.4f}')
    print(f'{tag}_n_trades:     {r["n_trades"]}')
    print(f'{tag}_n_rejects:    {r["n_rejects"]}')
    print(f'{tag}_active_days:  {r["active_days"]}')
    print(f'{tag}_win_days:     {r["win_days"]:.3f}')
    print(f'{tag}_n_markets:    {r["n_markets"]}')
    print(f'runtime_seconds:  {runtime:.1f}')
    if reason:
        print(f'constraint_violated: {reason}')


if __name__ == '__main__':
    d = load_data()
    n_val = int((d.expiry < VAL_SPLIT).sum())
    print(f'markets: {len(d.markets)} resolved ({n_val} val / {len(d.markets) - n_val} holdout)')
    print(f'price points: {sum(len(s[0]) for s in d.series)}')
    for ai, sym in enumerate(SYMS):
        k = d.binance[ai]
        print(f'{sym}: {len(k)} minutes  {datetime.fromtimestamp(k[0,0], tz=timezone.utc):%Y-%m-%d} '
              f'-> {datetime.fromtimestamp(k[-1,0], tz=timezone.utc):%Y-%m-%d}')
