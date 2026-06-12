#!/usr/bin/env python3
"""One-time data fetch for the Polymarket research domain.

Pulls into ~/.cache/autopoly/ :
  1. Market catalog for BTC/ETH daily "above $X" and "Up or Down" events
     (Gamma API, enumerated by deterministic event slugs per date).
  2. YES-token price history per market (CLOB prices-history API).
  3. Binance 1m klines for BTCUSDT/ETHUSDT (the markets' resolution source).
  4. Resolution outcomes (Gamma outcomePrices, cross-checked against the
     Binance candle rule stated in each market's resolution terms).

Run from repo root:   python3 polymarket/fetch_data.py [--quick]
--quick limits to the last 14 days of markets (smoke test).
Resume-safe: per-market price series are checkpointed to a jsonl file and
skipped on re-run; the final consolidated .npz files are rewritten.
"""
import json, os, re, sys, time, urllib.request
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

CACHE = os.path.expanduser('~/.cache/autopoly')
START = date(2026, 3, 1)    # first market expiry date to include
END = date(2026, 6, 12)     # last expiry date (inclusive, must be resolved)
BINANCE_START = date(2026, 2, 15)  # earlier, for vol warm-up
ASSETS = [('bitcoin', 'BTCUSDT'), ('ethereum', 'ETHUSDT')]
NY = ZoneInfo('America/New_York')
UA = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
          'august', 'september', 'october', 'november', 'december']


def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            sys.stderr.write(f'retry{i} {e} {url[:130]}\n')
            time.sleep(1.3 * (i + 1))
    return None


def iso_ts(s):
    s = re.sub(r'\.\d+', '', s).replace('Z', '+00:00')
    return datetime.fromisoformat(s).timestamp()


def fetch_catalog(start, end):
    """Enumerate daily events by slug, return list of market dicts."""
    markets, missing = [], []
    d = start
    while d <= end:
        mon, day = MONTHS[d.month - 1], d.day
        for ai, (aname, _) in enumerate(ASSETS):
            for kind, kname in [(0, f'{aname}-above-on-{mon}-{day}'),
                                (1, f'{aname}-up-or-down-on-{mon}-{day}')]:
                ev = None
                for slug in (f'{kname}-{d.year}', kname):
                    r = get(f'https://gamma-api.polymarket.com/events?slug={slug}')
                    if r:
                        ev = r[0]
                        break
                    time.sleep(0.05)
                if not ev:
                    missing.append(kname)
                    continue
                for m in ev.get('markets', []):
                    try:
                        toks = json.loads(m['clobTokenIds'])
                        op = json.loads(m.get('outcomePrices') or '[]')
                    except (KeyError, json.JSONDecodeError):
                        continue
                    resolved = op in (['1', '0'], ['0', '1'])
                    strike = np.nan
                    if kind == 0:
                        sm = re.search(r'above \$([\d,]+(?:\.\d+)?)', m.get('question', ''))
                        if not sm:
                            continue
                        strike = float(sm.group(1).replace(',', ''))
                    markets.append(dict(
                        cid=m['conditionId'], question=m.get('question', ''),
                        token_yes=toks[0], asset=ai, kind=kind, strike=strike,
                        expiry=iso_ts(m['endDate']), created=iso_ts(m['createdAt']),
                        fee=bool(m.get('feesEnabled')), tick=float(m.get('orderPriceMinTickSize') or 0.001),
                        date=d.isoformat(), resolved=resolved,
                        outcome_yes=(1 if op and op[0] == '1' else 0) if resolved else -1))
                time.sleep(0.08)
        d += timedelta(days=1)
    print(f'catalog: {len(markets)} markets, {len(missing)} event slugs missing', flush=True)
    if missing[:5]:
        print('  missing e.g.:', missing[:5], flush=True)
    return markets


def fetch_prices(markets):
    ckpt = os.path.join(CACHE, 'prices_ckpt.jsonl')
    done = {}
    if os.path.exists(ckpt):
        for line in open(ckpt):
            cid, arr = line.split('\t', 1)
            done[cid] = json.loads(arr)
        print(f'prices checkpoint: {len(done)} series loaded', flush=True)
    out = open(ckpt, 'a')
    for i, m in enumerate(markets):
        if m['cid'] in done:
            continue
        r = get(f"https://clob.polymarket.com/prices-history?market={m['token_yes']}&interval=max&fidelity=1")
        h = (r or {}).get('history') or []
        series = [[round(p['t']), round(p['p'], 4)] for p in h]
        done[m['cid']] = series
        out.write(m['cid'] + '\t' + json.dumps(series) + '\n')
        if i % 100 == 0:
            print(f'prices {i}/{len(markets)}', flush=True)
            out.flush()
        time.sleep(0.12)
    out.close()
    return done


def fetch_binance():
    res = {}
    for ai, (_, sym) in enumerate(ASSETS):
        rows = []
        t = int(datetime(BINANCE_START.year, BINANCE_START.month, BINANCE_START.day,
                         tzinfo=timezone.utc).timestamp() * 1000)
        now_ms = int(time.time() * 1000)
        while t < now_ms:
            d = get(f'https://data-api.binance.vision/api/v3/klines?symbol={sym}'
                    f'&interval=1m&startTime={t}&limit=1000')
            if not d:
                break
            rows += [[k[0] / 1000, float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in d]
            t = d[-1][0] + 60_000
            if len(d) < 1000:
                break
            time.sleep(0.1)
        arr = np.array(rows)
        # forward-fill onto a contiguous minute grid so index = (ts - t0) // 60
        t0, t1 = arr[0, 0], arr[-1, 0]
        grid = np.arange(t0, t1 + 60, 60)
        idx = np.searchsorted(arr[:, 0], grid, side='right') - 1
        full = arr[np.clip(idx, 0, len(arr) - 1)]
        full[:, 0] = grid
        np.savez_compressed(os.path.join(CACHE, f'binance_{sym}.npz'), k=full)
        res[ai] = full
        print(f'binance {sym}: {len(arr)} klines -> {len(full)} minute grid', flush=True)
    return res


def noon_et_close(k, d):
    """Close of the 1m candle opening at 12:00 ET on date d (the resolution rule)."""
    ts = datetime(d.year, d.month, d.day, 12, 0, tzinfo=NY).timestamp()
    i = int((ts - k[0, 0]) // 60)
    if 0 <= i < len(k) and abs(k[i, 0] - ts) < 1:
        return k[i, 4]
    return np.nan


def cross_check(markets, binance):
    mism, checked = 0, 0
    for m in markets:
        if not m['resolved']:
            continue
        d = date.fromisoformat(m['date'])
        k = binance[m['asset']]
        c = noon_et_close(k, d)
        if np.isnan(c):
            continue
        if m['kind'] == 0:
            b = 1 if c > m['strike'] else 0
        else:
            cp = noon_et_close(k, d - timedelta(days=1))
            if np.isnan(cp):
                continue
            b = 1 if c > cp else 0   # Up wins iff prev close < final close
        checked += 1
        if b != m['outcome_yes']:
            mism += 1
            if mism <= 5:
                print(f'  MISMATCH {m["question"]} gamma={m["outcome_yes"]} binance={b} close={c}', flush=True)
    print(f'outcome cross-check: {checked} checked, {mism} mismatches', flush=True)


def main():
    os.makedirs(CACHE, exist_ok=True)
    quick = '--quick' in sys.argv
    start = END - timedelta(days=13) if quick else START
    markets = fetch_catalog(start, END)
    json.dump(markets, open(os.path.join(CACHE, 'markets.json'), 'w'))
    binance = fetch_binance()
    cross_check(markets, binance)
    prices = fetch_prices(markets)
    ids, offsets, ts, ps = [], [0], [], []
    kept = 0
    for m in markets:
        s = prices.get(m['cid']) or []
        if len(s) < 3:
            continue
        ids.append(m['cid'])
        ts += [x[0] for x in s]
        ps += [x[1] for x in s]
        offsets.append(len(ts))
        kept += 1
    np.savez_compressed(os.path.join(CACHE, 'prices.npz'),
                        ids=np.array(ids), offsets=np.array(offsets, dtype=np.int64),
                        ts=np.array(ts, dtype=np.float64), ps=np.array(ps, dtype=np.float32))
    print(f'saved: {kept} price series, {len(ts)} points total -> {CACHE}', flush=True)


if __name__ == '__main__':
    main()
