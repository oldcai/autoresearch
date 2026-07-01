#!/usr/bin/env python3
"""The experiment file: a trading strategy for short-dated Polymarket crypto
binaries. This is the ONLY file the research agent modifies.

Baseline: price each market as a digital option off Binance realized
volatility; take liquidity when the market's printed price deviates from the
theoretical value by more than a threshold.

Run:  python3 polymarket/strategy.py > run.log 2>&1
then: grep "^val_sharpe:" run.log
"""
import math
import os

import numpy as np

from prepare import evaluate

# ----------------------- hyperparameters (edit me) -------------------------
VOL_LOOKBACK_MIN = 4320      # minutes of Binance history for realized vol
EDGE_THRESHOLD = 0.05        # required |theo - market| mispricing
ORDER_USD = 25.0             # notional per signal per step
MAX_POS_USD = 300.0          # max cost basis per market
PRICE_BAND = (0.03, 0.97)    # only act on prints inside this band
MONO_MARGIN = 0.02           # above-ladder monotonicity violation threshold
LADDER_RESID = 0.04          # trade strikes deviating this far from the ladder fit
LADDER_BUDGET = 100.0        # separate daily budget for ladder-RV trades
MONO_DAY_BUDGET = 100.0      # max arb notional per UTC day
FAVORITE_MIN = 0.70          # only buy a side already priced at least this
EXIT_EDGE = -0.05            # sell a held side when theo - market falls below this
REENTRY_BLOCK_MIN = 1080     # no re-entry this long after an edge-reversal exit
MAX_AGE_MIN = 30.0           # ignore prints staler than this
MIN_TTE_MIN = 60.0           # stop trading this close to expiry
VOL_FLOOR = 1e-5             # per-minute log-vol floor
REGIME_MAX = 2.0             # skip entries when 60min vol / lookback vol exceeds this
ASSET_CAP_USD = float(os.environ.get('ASSET_CAP_USD', '0'))  # max marked exposure per asset; 0 = off
THICK_MULT = float(os.environ.get('RUNNER_THICK_MULT', '2.4'))
THIN_POS_CAP_FRAC = float(os.environ.get('RUNNER_THIN_POS_CAP_FRAC', '0.25'))


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class Strategy:
    def __init__(self):
        self._vol_cache = {}
        self._blocked_until = {}
        self._arb_spend = {}

    def _sigma_per_min(self, obs, ai):
        key = (ai, int(obs.t // 300))
        if key not in self._vol_cache:
            c = obs.closes(ai, VOL_LOOKBACK_MIN)
            r = np.diff(np.log(np.maximum(c, 1e-9)))
            v_long = max(float(r.std()), VOL_FLOOR)
            v_short = max(float(r[-60:].std()), VOL_FLOOR) if len(r) > 60 else v_long
            self._vol_cache = {key: (v_long, v_short / v_long)}
        return self._vol_cache[key]

    def on_step(self, obs):
        orders = []
        equity = obs.cash + float(np.sum(obs.pos_yes * np.nan_to_num(obs.price, nan=0.5)
                                         + obs.pos_no * (1 - np.nan_to_num(obs.price, nan=0.5))))
        scale = max(1.0, equity / 10_000.0)
        sig = {ai: self._sigma_per_min(obs, ai)[0] for ai in (0, 1)}
        regime = {ai: self._sigma_per_min(obs, ai)[1] for ai in (0, 1)}
        spot = {ai: obs.spot(ai) for ai in (0, 1)}
        asset_exp = {}
        if ASSET_CAP_USD > 0:
            for jj in range(len(obs.idx)):
                p = obs.price[jj]
                if np.isnan(p):
                    continue
                ai = int(obs.asset[jj])
                asset_exp[ai] = asset_exp.get(ai, 0.0) + obs.pos_yes[jj] * p + obs.pos_no[jj] * (1 - p)

        def cap_buy(ai, usd):
            if ASSET_CAP_USD <= 0:
                return usd
            room = ASSET_CAP_USD - asset_exp.get(ai, 0.0)
            if room <= 0:
                return 0.0
            usd = min(usd, room)
            asset_exp[ai] = asset_exp.get(ai, 0.0) + usd
            return usd

        order_base = ORDER_USD * scale
        thin_pos_cap_usd = MAX_POS_USD * THIN_POS_CAP_FRAC

        def thin_cap_allows(j, usd):
            p = obs.price[j]
            if np.isnan(p):
                return False
            cost = obs.pos_yes[j] * p + obs.pos_no[j] * (1 - p)
            thick = usd >= THICK_MULT * order_base
            return thick or cost < thin_pos_cap_usd - 1e-9

        def buy_order(j, side, usd, limit):
            if not thin_cap_allows(j, usd):
                return 0.0
            usd = cap_buy(int(obs.asset[j]), usd)
            if usd <= 0:
                return 0.0
            orders.append((int(obs.idx[j]), side, usd, limit))
            return usd

        def buy_pair(a, b):
            if not (thin_cap_allows(a, ORDER_USD) and thin_cap_allows(b, ORDER_USD)):
                return 0.0
            ai = int(obs.asset[a])
            if ASSET_CAP_USD > 0:
                room = ASSET_CAP_USD - asset_exp.get(ai, 0.0)
                if room < 2 * ORDER_USD:
                    return 0.0
                asset_exp[ai] = asset_exp.get(ai, 0.0) + 2 * ORDER_USD
            orders.append((int(obs.idx[a]), 'BUY_YES', ORDER_USD,
                           min(obs.price[a] + 0.01, 0.99)))
            orders.append((int(obs.idx[b]), 'BUY_NO', ORDER_USD,
                           min(1 - obs.price[b] + 0.01, 0.99)))
            return 2 * ORDER_USD

        for j in range(len(obs.idx)):
            p_mkt = obs.price[j]
            if (np.isnan(p_mkt) or np.isnan(obs.strike[j])
                    or obs.age_min[j] > MAX_AGE_MIN
                    or obs.tte_min[j] < MIN_TTE_MIN
                    or not (PRICE_BAND[0] <= p_mkt <= PRICE_BAND[1])):
                continue
            ai = int(obs.asset[j])
            s, k = spot[ai], obs.strike[j]
            unstable = regime[ai] > REGIME_MAX
            vol = sig[ai] * math.sqrt(max(obs.tte_min[j], 1.0))
            d2 = (math.log(s / k) - 0.5 * vol * vol) / vol
            theo = _phi(d2)
            edge = theo - p_mkt
            # cut losers: edge has reversed against a held position
            if obs.pos_yes[j] > 0 and edge < EXIT_EDGE:
                orders.append((int(obs.idx[j]), 'SELL_YES',
                               obs.pos_yes[j] * p_mkt, None))
                self._blocked_until[int(obs.idx[j])] = obs.t + REENTRY_BLOCK_MIN * 60
                continue
            if obs.pos_no[j] > 0 and -edge < EXIT_EDGE:
                orders.append((int(obs.idx[j]), 'SELL_NO',
                               obs.pos_no[j] * (1 - p_mkt), None))
                self._blocked_until[int(obs.idx[j])] = obs.t + REENTRY_BLOCK_MIN * 60
                continue
            if self._blocked_until.get(int(obs.idx[j]), 0) > obs.t:
                continue
            cost = obs.pos_yes[j] * p_mkt + obs.pos_no[j] * (1 - p_mkt)
            if cost >= MAX_POS_USD or obs.cash < ORDER_USD:
                continue
            # favorite-side only: buy the high-probability side when the
            # model says it is still underpriced (sell tails, never buy them)
            if unstable:
                continue
            if obs.age_min[j] >= 5 and abs(edge) < 1.5 * EDGE_THRESHOLD:
                continue   # stale repeat without strong edge: skip churn
            if edge > EDGE_THRESHOLD and p_mkt >= FAVORITE_MIN:
                usd = scale * min(ORDER_USD * edge / EDGE_THRESHOLD, 4 * ORDER_USD)
                buy_order(j, 'BUY_YES', usd, min(p_mkt + 0.01, 0.99))
            elif -edge > EDGE_THRESHOLD and p_mkt <= 1 - FAVORITE_MIN:
                usd = scale * min(ORDER_USD * -edge / EDGE_THRESHOLD, 4 * ORDER_USD)
                buy_order(j, 'BUY_NO', usd, min(1 - p_mkt + 0.01, 0.99))
        # cross-strike consistency arb under a daily budget
        from collections import defaultdict
        day = int(obs.t // 86400)
        spent = self._arb_spend.get(day, 0.0)
        spent_l = self._arb_spend.get(('L', day), 0.0)
        if spent < MONO_DAY_BUDGET or spent_l < LADDER_BUDGET:
            groups = defaultdict(list)
            for j in range(len(obs.idx)):
                if (obs.kind[j] == 0 and not np.isnan(obs.price[j])
                        and obs.age_min[j] <= 5 and not np.isnan(obs.strike[j])):
                    groups[(int(obs.asset[j]), float(obs.expiry[j]))].append(j)
            for js in groups.values():
                js.sort(key=lambda j: obs.strike[j])
                for a, b in zip(js, js[1:]):
                    if spent >= MONO_DAY_BUDGET:
                        break
                    if obs.price[a] + MONO_MARGIN < obs.price[b] and obs.cash > 2 * ORDER_USD:
                        spent += buy_pair(a, b)
                # ladder-implied relative value: isotonic (decreasing) fit
                if len(js) >= 4 and spent_l < LADDER_BUDGET:
                    ps = np.array([obs.price[j] for j in js])
                    fit = np.minimum.accumulate(np.maximum.accumulate(ps[::-1])[::-1])
                    # average of forward cummin and backward cummax envelopes
                    fit = 0.5 * (fit + np.maximum.accumulate(ps[::-1])[::-1])
                    for j, pj, fj in zip(js, ps, fit):
                        if spent_l >= LADDER_BUDGET or obs.cash < ORDER_USD:
                            break
                        r = fj - pj
                        if r > LADDER_RESID and pj >= FAVORITE_MIN:
                            spent_l += buy_order(j, 'BUY_YES', ORDER_USD, min(pj + 0.01, 0.99))
                        elif -r > LADDER_RESID and pj <= 1 - FAVORITE_MIN:
                            spent_l += buy_order(j, 'BUY_NO', ORDER_USD, min(1 - pj + 0.01, 0.99))
            self._arb_spend[day] = spent
            self._arb_spend[('L', day)] = spent_l
        return orders


if __name__ == '__main__':
    evaluate(Strategy)
