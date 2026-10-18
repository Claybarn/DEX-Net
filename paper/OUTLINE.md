# Paper outline — arXiv preprint

Working title: **Learning Only the Stochastic Part: A Mechanics-Grounded Generative Model of Uniswap v3 Order Flow**

Alternatives: *Neural Event Simulation for Concentrated-Liquidity AMMs*; *Simulating Uniswap v3 Pools with Exact Mechanics and Learned Order Flow*.

Status legend: **[have]** evidence exists in the repo (`paper/results/`) or on the GPU box · **[partial]** data exists, analysis or figure missing · **[run]** new experiment needed.

All model results come from the final run matrix (`paper/final/matrix.json`):

- 5 model families × 3 seeds (260930–260932), trained with identical code, schedules and hardware (4× A100-SXM4-80GB).
- Results are in `paper/results/final/`.
- Rollouts use 1,217 fixed anchors: 610 test and 607 validation; 3 seeds × 32 paths at 32/128/512 events.

---

## Abstract (draft)

Liquidity providers in concentrated-liquidity AMMs need simulators that reproduce how swaps, mints and burns arrive and interact, not just price paths.

We present a pooled neural event model for Uniswap v3. It learns only the stochastic parts of order flow (timing, event type, side, size relative to available depth, liquidity ranges and amounts) and computes prices and liquidity exactly from pool state. Conditioning the model on the pool's liquidity profile around the current price, which the exact mechanics provide at no cost during simulation, is what makes long rollouts work.

The model is trained on 29.4 million events from ten Ethereum pools, with active liquidity reconciled exactly at all 28.2 million swaps. On pools held out from training, its 512-event rollouts:
- halve the energy score of an otherwise identical model without the liquidity profile;
- beat history-based baselines at every horizon;
- cut spurious liquidity collapse from 24% to 15% of paths (real: 5%);
- get the width of the price distribution right (1.01× realised, against 1.78× without the profile).

The liquidity-aware model has slightly *worse* next-event likelihood: one-step likelihood and simulation quality are different targets. We document the remaining failure modes. Predicted skew is not directionally informative, swap-size tails are still too heavy, and JIT liquidity is too large relative to its swaps.

Finally, we measure what an LP simulator must get right. Across 16 pools on Ethereum, Arbitrum and Base, fees roughly equal loss-versus-rebalancing (LVR), so value comes from timing. A perfect-foresight bound shows that this value concentrates at decision intervals of minutes rather than days.

We release code and a reconciled dataset.

---

## 1. Introduction

- **Motivation:** LP P&L is fees minus LVR; both depend on the joint dynamics of order flow and liquidity, not on price alone.
- **The gap:** price-only simulators (GBM, GARCH, return generators) ignore liquidity; agent-based models need hand-specified behaviour.
- **The idea:** split the deterministic (AMM mechanics) from the stochastic (who trades or provides liquidity, when, and how much). Learn the second and compute the first exactly. The exact state also gives the model information a sequence model cannot infer reliably: the liquidity profile around the price.
- **Contributions:**
  1. A reconciled multi-pool v3 event dataset with exact state checks.
  2. A pooled event model with mechanics-consistent sampling and liquidity-profile conditioning.
  3. A multi-seed evaluation: likelihood, rollout fidelity against baselines, liquidity behaviour, and documented failure modes, including the gap between likelihood and simulation quality.
  4. An LVR-based economic lens on what simulators must capture, with cross-chain measurements and a timing ceiling.

## 2. Background and related work

- **Uniswap v3 mechanics:** ticks, active liquidity, the swap step, mint/burn on ranges, fee growth, the protocol fee.
- **LVR and LP economics:** Milionis et al. (LVR), arbitrage with fees and discrete blocks, empirical LP returns, JIT liquidity.
- **Neural temporal point processes:** neural Hawkes, Transformer Hawkes, intensity-free models.
- **Generative market simulators:** limit-order-book generators (token or sequence models, GAN and diffusion approaches) and agent-based models (e.g. ABIDES), plus the stylized-facts evaluation tradition.

## 3. Data

- **Collection:** 10 Ethereum v3 pools (USDC, USDT, DAI and WBTC paired with WETH at 0.05%/0.3%, plus USDC/USDT at 0.01%/0.05%), selected by existence at a fixed roster block, not current activity. Covers May 2021 to September 2026; Initialize, Swap, Mint and Burn logs. **[have]**
- **Reconciliation:**
  - an exact-integer Fenwick ledger reproduces active liquidity at all 28,241,248 swaps;
  - the header hash chain is checked;
  - owner and range balances stay non-negative;
  - 516,415 burn-support checks pass. **[have]**
- **Representation:** economic base/quote orientation; sizes relative to virtual reserves; liquidity relative to active liquidity; ranges relative to the current tick; pool metadata (fee, tick spacing, decimals). **[have]**
- **Liquidity profile:** at every event, the active liquidity at 17 log-price offsets from −4% to +4%, as log1p(L(offset)/L(now))/4. Training reads it from the Fenwick ledger (`scripts/prepare_depth.py`); rollouts compute it from the simulated book. The two agree exactly. **[have]**
- **Splits:** chronological train/val/test by block, with purged horizons; 8 fitting pools and 2 held-out pools (WBTC/WETH 0.3%, USDC/USDT 0.05%); scalers fit on training rows only. **[have]**
- **Supplementary data for §6:** 61M verified swaps from the six WETH/USDC pools on Arbitrum and Base (September 2025 to September 2026; 156/156 on-chain price checkpoints match) and Binance ETHUSDT 1-second prices. **[have]**
- **Table 1:** per-pool event counts, date ranges and split sizes. **[partial]** The numbers exist; the table needs writing.

## 4. Model

- **Event space:** sell, buy, mint, burn. The clock has a same-block continuation logit plus a geometric mixture over block gaps.
- **Marks:**
  - swap size: a rational-quadratic spline flow on log(base / virtual reserve), with buys truncated at exact capacity;
  - mint range: discretized logistic mixtures on the tick grid (lower bound and width);
  - mint amount: a flow on log(L / active L);
  - burn: range restricted to existing inventory, a full-burn Bernoulli, and a partial-fraction flow.
- **Encoder:** a Transformer over the last 64 events (12 features plus masks) with a metadata query token.
  - The headline model is depth-aware: the liquidity profile is projected and added to the query token, at d=128, 4 layers and about 0.95M parameters.
  - The reference model is identical except for the profile. The small model is 64×2.
- **Mechanics:** exact tick-crossing swap step and liquidity updates. Invalid sampled events become explicit absorbing failures, never silently fixed. **[have]**
- **Training:**
  - next-event NLL, pool-balanced over 3 epochs (202M examples), AdamW, 1,024-example updates;
  - checkpoint selected by seen-pool validation macro NLL (fixed budget, no early stopping);
  - sync-free static loss path;
  - interval edges of the discretized range likelihood computed in float64 (in float32, a one-tick interval far from the anchor can vanish below one ulp);
  - about 11–14 h per run with 15 runs sharing 4 A100s, about 5k examples/s per 128×4 run. **[have]**
- **Figure 1:** architecture and sampling loop, showing the learned versus deterministic split and where the liquidity profile enters. **[run]** (drawing only)

## 5. Evaluation

### 5.1 Next-event likelihood **[have]** `paper/results/final/nll.json`

Macro NLL in nats/event, mean ± s.d. over 3 seeds; 32,768 fixed examples per pool:

| Model | Seen pools | Held-out pools |
|---|---|---|
| Reference 128×4 | 5.534 ± 0.005 | 7.649 ± 0.003 |
| Depth-aware 128×4 | 5.542 ± 0.005 | 7.667 ± 0.019 |
| Depth-aware 64×2 | 5.620 ± 0.010 | 7.797 ± 0.028 |
| Depth, last 8 events only | 5.722 ± 0.004 | 7.953 ± 0.046 |
| Depth, no history | 6.583 ± 0.024 | 23.6 ± 9.8 |

- **The liquidity profile costs a little likelihood:** +0.008 seen (s.e. 0.004) and +0.018 held-out (s.e. 0.011). This sets up §5.4.
- **Capacity:** the 64×2 model costs +0.08 / +0.13.
- **History is essential:**
  - an 8-event window costs +0.18 / +0.29, far beyond seed noise;
  - without history, seen NLL rises by +1.04 and held-out timing fails;
  - the held-out failure is almost entirely the wait head on USDC/USDT 0.05% (wait NLL 12–47 against about 1–2 normally). Without history the model cannot tell how active an unseen pool is, and validation selects very early checkpoints (rounds 2, 3 and 16).
- **Missing baselines:** an unconditional per-pool model (empirical marks, renewal clock) and a neural Hawkes / Transformer Hawkes model on the same event space. **[run]**
- **Table 2:** NLL by pool and head with baselines. **[partial]** Per-pool and per-head numbers are in `nll.json`.

### 5.2 Rollout fidelity **[have]** `paper/results/final/rollout-summaries/`, `compare-*.json`

Energy score over four endpoints (return, liquidity change, elapsed blocks, volume); lower is better. Test anchors; models are the mean of 3 seeds:

| Pools | Events | Reference | **Depth-aware** | Depth 64×2 | Recent baseline | Expanding baseline |
|---|---|---|---|---|---|---|
| Held-out | 32 | 3.43 | **2.75** | 3.03 | 3.15 | 3.61 |
| Held-out | 128 | 8.79 | **6.38** | 6.34 | 6.48 | 7.23 |
| Held-out | 512 | 37.24 | **16.55** | 16.76 | 16.83 | 17.64 |
| Seen | 32 | 3.01 | **3.00** | 3.01 | 3.57 | 4.10 |
| Seen | 128 | 7.82 | 7.88 | 8.23 | 8.00 | 10.82 |
| Seen | 512 | 26.49 | 25.71 | 31.46 | **21.93** | 32.73 |

- **Head-to-head, depth minus reference** (paired bootstrap over anchors):
  - held-out 512 events: −18.5 [−27.3, −10.8] on test and −19.4 [−36.6, −5.6] on validation;
  - held-out 128 events: −2.6 [−4.4, −1.2];
  - seen pools: tie or small gain, e.g. −1.34 [−2.59, −0.17] at 512 on validation.
- **Validation anchors (seen pools):** the depth-aware model beats both baselines at every horizon (512 events: 37.95 vs recent 44.16).
  - On the seen-pool *test* period, the recent baseline still wins at 512 events. This is a period effect, and it should be stated.
- **Liquidity collapse** (share of paths losing more than 63% of active liquidity), held-out, 512 events: reference 24.2%, depth 15.2%, real 5.1%.
- **Price-spread ratio** (5–95% width, simulated over realised), held-out, 512 events: reference 1.78×, depth 1.01×.
- **Capacity:** the 64×2 depth model matches 128×4 on held-out pools but is clearly worse on seen pools at long horizons (+5.3 [4.4, 6.4] at 512).
- **Path reweighting** (fitted on seen-pool validation rollouts only), `reweight-depth.json`:
  - 512-event energy improves by −1.65 [−2.18, −1.19] on seen pools and −1.92 [−3.23, −0.92] on held-out pools;
  - reweighted, the model beats the recent baseline on held-out pools at 512 (17.35 vs 17.57) but not on seen pools (27.2 vs 22.6).
  - Present it as an optional calibration layer, not part of the model.
- **Calibration:** PIT histograms and outer-decile frequencies by horizon, `calibration-depth.json`. **[have]** Needs a figure.
- **Asymmetry:** predicted skew is not directionally informative, even when the model is most confident.
  - Skew–tail correlation is −0.12 to +0.0 with CIs spanning 0.
  - On held-out pools, confident down-skew forecasts were followed by upside breakouts 65–86% of the time. The model carries recent momentum into its skew, while realised moves revert slightly.
  - Report this as a limitation. `skew-depth.json` **[have]**
- **Stylized facts:** return and |return| ACF, volatility against horizon, inter-arrival and same-block clustering, side persistence, event mix; simulated vs real. `stylized-facts-depth.json`, `stylized-facts-reference.json` **[have]** The table and figure are still to write.
- **Direct (non-recursive) forecaster:** include as the "static cone" contrast. Existing seeds are in `paper/results/` from the earlier evaluation. **[partial]** It needs scoring on the final anchors.
- **Figure 2:** forecast cones for several origins with real paths overlaid, depth vs reference. **[partial]**
- **Figure 3:** stylized-facts panel. **[partial]**

### 5.3 Liquidity behaviour

- **Swap-size tail and liquidity collapse:** these explain the reference model's long-horizon failure. `swap-diagnostics-*.json` **[have]**
  - One-step size calibration is fine; the error is state-dependent. Once simulated liquidity thins, swaps sized relative to the virtual reserve move the price far more.
  - Held-out pools, share of swaps moving the price more than 100 bp: reference 0.97%, depth 0.20%, real 0.03%.
  - More than 500 bp: reference 0.48%, depth 0.066%, real 0.018%.
  - The depth-aware model removes most of the excess but not all of it.
- **Mint and burn heads:** NLL by head (§5.1).
- **JIT-like sequences** (earlier model; to re-check on the final depth model):
  - The association is learned: the swap accompanying a mint is much larger than ordinary swaps (median 58,400 vs 3,599 USDC in USDC/WETH). **[have]**
  - Failure mode: generated swap size relative to deposited capital is too large (median 1.99% vs 0.11%). **[have]**
  - Failure mode: large mints stay unremoved after 64 events in 16–34% of generated cases against 0.28% historically. **[have]**
  - **[run]** Repeat both on the final depth-aware rollouts.
- **Figure 4:** JIT scatter (deposit vs swap, generated vs historical) and a removal-survival curve. **[partial]**

### 5.4 Likelihood is not simulation quality **[have]**

The clearest evidence is now the final matrix itself. The depth-aware model is slightly *worse* in next-event NLL (+0.018 held-out), yet halves held-out 512-event energy (37.2 → 16.6) and removes most spurious liquidity collapse.

Supporting evidence:
- The 64×2 model is much worse in NLL but matches 128×4 in held-out rollouts.
- Earlier retained-history models: NLL improved (5.336 vs 5.346), but price CRPS at 128–512 events was 7–10% worse.

Explanation: the failures that matter for simulation are rare, state-dependent events (thin liquidity), which weigh little in average one-step likelihood but compound over a rollout.

- **Figure 5:** NLL against 512-event energy for all model families with seed error bars. **[partial]** The data is in `paper/results/final/`.

## 6. An economic lens: what must an LP simulator get right?

- **The decomposition:** LP P&L = fees − LVR, measured per swap against Binance 1-second prices. This is exact for a small in-range position. **[have]** `fee_lvr.py`, validated against σ²/8 on synthetic pools.
- **Fees ÷ LVR by pool**, over one year (September 2025 to September 2026):

  | Pool | Fees ÷ LVR |
  |---|---|
  | Ethereum USDC/WETH 0.05% | 0.77 |
  | Ethereum USDT/WETH 0.3% | 1.00 |
  | Base 0.05% | 1.00 |
  | Base 0.3% | 1.06 |
  | Arbitrum 0.05% | 0.92 |
  | Arbitrum 0.01% | 0.47 |

  LVR is about 1.3 bp/day everywhere, so competition pushes fees toward LVR. Five-year mainnet history is also available, along with the effect of the December 2025 protocol-fee switch. **[have]**
- **Timing ceiling:** perfect-foresight in/out value by decision interval (block, 1 min, 5 min, 1 h, 1 d) with entry costs. Value concentrates at minutes, which motivates event-level simulators over horizon forecasters. **[have]** `ceiling.py`
- **Simulator test:** do model rollouts reproduce the conditional distribution of fees − LVR over the next interval, not just price? Score fees − LVR energy and coverage from rollouts against realized values at 5 min and 1 h. This is the bridge between §5 and §6. **[run]**
- **Figure 6:** fees ÷ LVR across 16 pools and three chains with intervals.
- **Figure 7:** timing ceiling against decision interval.
- **Not claimed:** profitable trading. State plainly that no forecast-driven policy beat costs in our tests.

## 7. Limitations

- Single-pool histories: there is no cross-pool or CEX information in the model.
- Small-position overlay: our liquidity doesn't affect the flow.
- Ethereum-only training; two held-out pools.
- Skew is not directionally informative; momentum is extrapolated.
- Swap-size tails are still heavier than real ones; some spurious liquidity collapse remains (15% vs 5%).
- JIT size mismatch (to be re-checked on the depth-aware model).
- On the seen-pool test period, a recent-history baseline beats the model at 512 events.

## 8. Conclusion

Learning only the stochastic part, combined with exact mechanics, gives a faithful and inspectable AMM simulator. The mechanics are also an input: conditioning on the exact liquidity profile is what makes long rollouts work on unseen pools. Next-event likelihood is the wrong target for long-horizon simulation. Multi-pool and CEX-conditioned simulation is the natural next step.

## Appendices

- A. Reconciliation algorithm and checks.
- B. Feature and target definitions; scaler protocol; liquidity-profile definition.
- C. Hyperparameters, compute and the final run matrix; float64 interval edges.
- D. Mechanics edge cases: capacity truncation, uint128 limits, burn support, failure handling.
- E. LVR accounting derivation and synthetic validation.
- F. L2 data collection and verification.
- G. Seed-level tables for all model families.

---

## Work still to run, in priority order

1. **Figures 1–7 and Tables 1–2** from `paper/results/final/` (no new compute).
2. **JIT analyses on the final depth-aware rollouts** (CPU, from saved traces).
3. **Next-event baselines:** unconditional per-pool model and a Hawkes / Transformer Hawkes model (one GPU run each).
4. **Direct forecaster scored on the final anchors** (CPU).
5. **Fees − LVR distribution test** from rollouts (§6 bridge).
6. **Optional:** a fresh October 2026 held-out period.

## Repository status

The code lives in this repository: `dexnet/` (library) and `scripts/` (stages). The event model, data pipeline, direct baseline and economics scripts were extracted from the research codebase without behavioural changes; they reproduced the original NLLs, objectives and economics outputs exactly before the later changes below.

Later changes:
- depth-aware model;
- fused and static training paths;
- fast validation;
- float64 range-interval edges, which changes reference-model NLL by about 2e-6.

Decided:

- Title "DEX-Net"; author Clayton Barnes.
- Apache-2.0 for code; CC BY 4.0 for released data.
- Dataset released.
- Direct model included as a baseline.
- Depth-aware 128×4 is the headline model; reference, small and history ablations are reported with 3 seeds each.

Reproducing the final results:

```
python -m scripts.run_matrix --matrix paper/final/matrix.json --data … --context … --depth … --out runs/final --gpus 0,1,2,3
RUNS=runs/final ANCHORS=eval DATA=… OUT=eval/final bash paper/final/evaluate.sh
```

Superseded preliminary results (single-seed or provisional runs) remain in `paper/results/*.json` outside `final/`. They are kept for the record and should not be cited.
