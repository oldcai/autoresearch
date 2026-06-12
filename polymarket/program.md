# autopoly — Polymarket strategy autoresearch

The autoresearch loop (see the root `program.md` for the original LLM-training
edition) pointed at a new domain: discovering profitable trading strategies
for Polymarket's short-dated crypto binary markets, in a fixed offline
backtest. Same contract, different metric.

## Setup

1. **Agree on a run tag** (e.g. `jun13-poly`). Branch `autoresearch/<tag>`
   must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from master.
3. **Read the in-scope files**:
   - `polymarket/README.md` — domain context, objective rationale, data notes.
   - `polymarket/prepare.py` — constants, data loading, backtest engine,
     evaluation. Do not modify.
   - `polymarket/strategy.py` — the file you modify.
4. **Verify data exists**: `~/.cache/autopoly/` must contain `markets.json`,
   `prices.npz`, `binance_*.npz`. If not, tell the human to run
   `python3 polymarket/fetch_data.py`.
5. **Initialize `polymarket/results.tsv`** with just the header row. The
   baseline is recorded after the first run.
6. **Confirm and go.**

## Experimentation

Each experiment is a single CPU backtest with a **fixed wall-clock budget of
5 minutes** (enforced by `evaluate()`). Launch:

```
python3 polymarket/strategy.py > run.log 2>&1
grep "^val_sharpe:\|^constraint_violated:" run.log
```

**The goal: get the highest `val_sharpe`.** It is the annualized Sharpe of
daily mark-to-market PnL on a fixed $10k bankroll over the validation window,
under the conservative taker-only execution model in `prepare.py`.

**What you CAN do** — anything inside `polymarket/strategy.py`: pricing
models (vol estimators, jump/fat-tail adjustments, time-of-day effects,
up-or-down vs above differences), signal thresholds, sizing, inventory and
hedging rules, market selection filters, early exits via SELL orders.

**What you CANNOT do:**
- Modify `polymarket/prepare.py` or `polymarket/fetch_data.py`.
- Install packages (numpy/pandas/stdlib only).
- Touch the HOLDOUT split: never set `EVAL_HOLDOUT=1` and never condition on
  data after the VAL_SPLIT date in any way. The human runs the holdout at
  milestones; a strategy that wins val but fails holdout is overfit — prefer
  fewer, more principled parameters.
- Trade live. This domain is offline backtesting only.

**Hard constraints** (violations score 0): max drawdown ≤ 25%, ≥ 30 fills,
≥ 15 active days, runtime ≤ 5 min. A high Sharpe earned by selling tails and
surviving by luck will show up as drawdown when it doesn't — treat the
constraint as part of the objective, not an obstacle.

**Simplicity criterion**: as in the original program — all else equal,
simpler is better; deleting code for equal `val_sharpe` is a win.

**The first run** establishes the baseline: run `strategy.py` as is.

## Logging results

`polymarket/results.tsv`, tab-separated, header plus 5 columns:

```
commit	val_sharpe	max_dd	status	description
a1b2c3d	0.412300	0.0820	keep	baseline: realized-vol digital pricing, edge>0.05
```

Use 0.000000 / 0.0 for crashes. Do not commit results.tsv.

## The experiment loop

LOOP FOREVER (identical contract to the root `program.md`):

1. Look at git state.
2. Modify `polymarket/strategy.py` with one experimental idea.
3. `git commit`.
4. `python3 polymarket/strategy.py > run.log 2>&1`
5. `grep "^val_sharpe:" run.log` — empty grep = crash; read the tail, fix if
   trivial, else log `crash` and move on.
6. Improved (strictly higher `val_sharpe`) → keep the commit. Worse or equal →
   `git reset --hard HEAD^` back to where you started.
7. Append the row to `polymarket/results.tsv` (untracked).
8. If a run exceeds 10 minutes, kill it and treat as failure.

**NEVER STOP.** Do not ask whether to continue. If out of ideas: re-read the
reference-trader findings in `polymarket/README.md` (where the real edges
were: stale prints near expiry, post-crash insurance premia, favorite side
accumulation, fee-aware price extremes), then try the next idea.
