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

import numpy as np

from prepare import evaluate

# ----------------------- hyperparameters (edit me) -------------------------
VOL_LOOKBACK_MIN = 720       # minutes of Binance history for realized vol
EDGE_THRESHOLD = 0.05        # required |theo - market| mispricing
ORDER_USD = 25.0             # notional per signal per step
MAX_POS_USD = 150.0          # max cost basis per market
PRICE_BAND = (0.03, 0.97)    # only act on prints inside this band
FAVORITE_MIN = 0.70          # only buy a side already priced at least this
EXIT_EDGE = -0.05            # sell a held side when theo - market falls below this
MAX_AGE_MIN = 30.0           # ignore prints staler than this
MIN_TTE_MIN = 10.0           # stop trading this close to expiry
VOL_FLOOR = 1e-5             # per-minute log-vol floor


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class Strategy:
    def __init__(self):
        self._vol_cache = {}

    def _sigma_per_min(self, obs, ai):
        key = (ai, int(obs.t // 300))
        if key not in self._vol_cache:
            c = obs.closes(ai, VOL_LOOKBACK_MIN)
            r = np.diff(np.log(np.maximum(c, 1e-9)))
            self._vol_cache = {key: max(float(r.std()), VOL_FLOOR)}
        return self._vol_cache[key]

    def on_step(self, obs):
        orders = []
        sig = {ai: self._sigma_per_min(obs, ai) for ai in (0, 1)}
        spot = {ai: obs.spot(ai) for ai in (0, 1)}
        for j in range(len(obs.idx)):
            p_mkt = obs.price[j]
            if (np.isnan(p_mkt) or np.isnan(obs.strike[j])
                    or obs.age_min[j] > MAX_AGE_MIN
                    or obs.tte_min[j] < MIN_TTE_MIN
                    or not (PRICE_BAND[0] <= p_mkt <= PRICE_BAND[1])):
                continue
            ai = int(obs.asset[j])
            s, k = spot[ai], obs.strike[j]
            vol = sig[ai] * math.sqrt(max(obs.tte_min[j], 1.0))
            d2 = (math.log(s / k) - 0.5 * vol * vol) / vol
            theo = _phi(d2)
            edge = theo - p_mkt
            # cut losers: edge has reversed against a held position
            if obs.pos_yes[j] > 0 and edge < EXIT_EDGE:
                orders.append((int(obs.idx[j]), 'SELL_YES',
                               obs.pos_yes[j] * p_mkt, None))
                continue
            if obs.pos_no[j] > 0 and -edge < EXIT_EDGE:
                orders.append((int(obs.idx[j]), 'SELL_NO',
                               obs.pos_no[j] * (1 - p_mkt), None))
                continue
            cost = obs.pos_yes[j] * p_mkt + obs.pos_no[j] * (1 - p_mkt)
            if cost >= MAX_POS_USD or obs.cash < ORDER_USD:
                continue
            # favorite-side only: buy the high-probability side when the
            # model says it is still underpriced (sell tails, never buy them)
            if edge > EDGE_THRESHOLD and p_mkt >= FAVORITE_MIN:
                usd = min(ORDER_USD * edge / EDGE_THRESHOLD, 4 * ORDER_USD)
                orders.append((int(obs.idx[j]), 'BUY_YES', usd,
                               min(p_mkt + 0.02, 0.99)))
            elif -edge > EDGE_THRESHOLD and p_mkt <= 1 - FAVORITE_MIN:
                usd = min(ORDER_USD * -edge / EDGE_THRESHOLD, 4 * ORDER_USD)
                orders.append((int(obs.idx[j]), 'BUY_NO', usd,
                               min(1 - p_mkt + 0.02, 0.99)))
        return orders


if __name__ == '__main__':
    evaluate(Strategy)
