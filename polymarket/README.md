# autopoly: Polymarket strategy research domain

A second research domain for this repo's autoresearch loop: instead of
minimizing `val_bpb` on a 5-minute GPT training run, the agent maximizes
`val_sharpe` on a 5-minute offline backtest of Polymarket's short-dated
crypto binary markets. The original LLM domain (root `prepare.py` /
`train.py`) is untouched.

## Why this domain

Built from a deep analysis (2026-06-12) of Polymarket account
`0x06dc51826bc524d9a83770e7de9dd7e005b04524`, an automated dealer in BTC/ETH/
SOL/XRP short-dated digitals: 80 days, ~$1.27M net deposits, +$402.6k trading
PnL (Sharpe 3.56, skew −2.26, worst day −$160k in the June 2–6 crash) plus
~$387k in platform incentives. Mechanics that make the domain tractable:

- Markets resolve **mechanically off Binance 1m candles** ("above $K" = noon
  ET close; "Up or Down" = noon-vs-noon close; terms in each market's
  description). Fair value is computable from a public free feed.
- Daily ladders for BTC/ETH exist every day (~14 strikes/day/asset plus one
  up-or-down), created ~6–7 days before expiry — thousands of independent,
  short-horizon pricing problems.
- Since 2026-03-06 crypto markets charge takers `0.07·p(1−p)` (makers free,
  20% recycled to makers) — fees are smallest exactly where stale quotes
  near expiry live.
- Documented edge sources: stale prints when Binance moves (the account's
  fills cluster before the 12:00 ET resolution), retail favorite-longshot
  bias (Bloomberg 2026-04-28: >100k wallets lost ≥$1k, ~2:1 losers:winners),
  post-crash insurance premia, spread capture via two-sided NO/YES bids.

## Objective (fixed in `prepare.py`)

`val_sharpe` — annualized Sharpe of daily MTM PnL, $10k bankroll, validation
window = markets expiring **before 2026-05-15**. Markets expiring on/after
that date are HOLDOUT, evaluated by the human only (`EVAL_HOLDOUT=1`), never
by the loop. Hard constraints: maxDD ≤ 25%, ≥30 fills, ≥15 active days.
Sharpe alone flatters short-vol strategies (the reference account's −2.26
skew); the drawdown constraint and the held-out June crash exist to catch
that.

## Execution model honesty (read before trusting any number)

Deliberately conservative, but still a model:

- **Taker-only**: fills happen only at the market's next *historically
  recorded* print after order submission, worsened by 1 tick, plus the
  `0.07·p(1−p)` fee where the market had fees enabled. No maker fills, no
  rebates — the reference account got ~39% maker fills, so this
  understates the opportunity rather than overstating it.
- Price series are the CLOB `prices-history` chart data (sparse, ~6–10
  prints/hour): fewer fill opportunities than reality.
- $100 max notional per market per step (thin books).
- Not modeled: book depth, queue position, latency races, liquidity rewards,
  multi-asset margin. **A good val_sharpe here is a candidate, not an edge** —
  Phase 2 (paper trading against the live CLOB) is the real test.

## Files

| file | role |
|---|---|
| `fetch_data.py` | one-time data pull (human runs; ~15 min full window) |
| `prepare.py` | constants + data + backtest engine + `evaluate()` — read-only |
| `strategy.py` | the experiment file — the only thing the agent edits |
| `program.md` | the autoresearch loop contract for this domain |

Data cache: `~/.cache/autopoly/` (markets.json, prices.npz, binance_*.npz).
Refresh/extend by editing the date constants in `fetch_data.py` and re-running
(checkpointed, resume-safe).

## Roadmap

- **Phase 1 (this)**: offline loop, maximize val_sharpe, holdout checks at
  milestones.
- **Phase 2**: paper trading — replay the winning strategy against the live
  order book via websocket for 2–4 weeks; calibrate the fill model gap.
- **Phase 3**: small live capital ($5–10k) behind hard limits (≤1% per
  market, ≤50% deployed, daily-loss kill switch). Jurisdiction/ToS review
  required before any live order.
