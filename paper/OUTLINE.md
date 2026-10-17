# Paper outline — arXiv preprint

Working title: **Learning Only the Stochastic Part: A Mechanics-Grounded Generative Model of Uniswap v3 Order Flow**

Alternatives: *Neural Event Simulation for Concentrated-Liquidity AMMs*; *Simulating Uniswap v3 Pools with Exact Mechanics and Learned Order Flow*.

Status legend: **[have]** evidence exists in the current repo or on the GPU box · **[partial]** data exists, analysis or figure missing · **[run]** new experiment needed.

---

## Abstract (draft)

Liquidity providers in concentrated-liquidity AMMs need simulators that reproduce how swaps, mints and burns arrive and interact, not just price paths.

We present a pooled neural event model for Uniswap v3. It learns only the stochastic parts of order flow (timing, event type, side, size relative to available depth, liquidity ranges and amounts) and computes prices and liquidity exactly from pool state.

The model is trained on 29.4 million events from ten Ethereum pools, with active liquidity reconciled exactly at all 28.2 million swaps. It improves next-event likelihood on seen and held-out pools, and its rollouts reproduce state-dependent, asymmetric price distributions and JIT-like liquidity behaviour. We document clear failure modes, including the size of JIT liquidity relative to the swaps it accompanies, and show that better one-step likelihood does not imply better multi-step simulation.

Finally, we measure what an LP simulator must get right. Across 16 pools on Ethereum, Arbitrum and Base, fees roughly equal loss-versus-rebalancing (LVR), so value comes from timing. A perfect-foresight bound shows that this value concentrates at decision intervals of minutes rather than days.

We release code and a reconciled dataset.

---

## 1. Introduction

- **Motivation:** LP P&L is fees minus LVR; both depend on the joint dynamics of order flow and liquidity, not on price alone.
- **The gap:** price-only simulators (GBM, GARCH, return generators) ignore liquidity; agent-based models need hand-specified behaviour.
- **The idea:** split the deterministic (AMM mechanics) from the stochastic (who trades or provides liquidity, when, and how much). Learn the second and compute the first exactly.
- **Contributions:**
  1. A reconciled multi-pool v3 event dataset with exact state checks.
  2. A pooled event model with mechanics-consistent sampling.
  3. An evaluation: likelihood, rollout fidelity, liquidity behaviour, and failure modes.
  4. An LVR-based economic lens on what simulators must capture, with cross-chain measurements and a timing ceiling.

## 2. Background and related work

- **Uniswap v3 mechanics:** ticks, active liquidity, the swap step, mint/burn on ranges, fee growth, the protocol fee.
- **LVR and LP economics:** Milionis et al. (LVR), arbitrage with fees and discrete blocks, empirical LP returns, JIT liquidity.
- **Neural temporal point processes:** neural Hawkes, Transformer Hawkes, intensity-free models.
- **Generative market simulators:** limit-order-book generators (token or sequence models, GAN and diffusion approaches) and agent-based models (e.g. ABIDES), plus the stylized-facts evaluation tradition.

## 3. Data

- **Collection:** 10 Ethereum v3 pools (USDC, USDT, DAI and WBTC paired with WETH at 0.05%/0.3%, plus USDC/USDT at 0.01%/0.05%), selected by existence at a fixed roster block, not current activity. Covers May 2021 to September 2026; Initialize, Swap, Mint and Burn logs. **[have]** `reports/pooled-data-20260930`
- **Reconciliation:**
  - an exact-integer Fenwick ledger reproduces active liquidity at all 28,241,248 swaps;
  - the header hash chain is checked;
  - owner and range balances stay non-negative;
  - 516,415 burn-support checks pass. **[have]**
- **Representation:** economic base/quote orientation; sizes relative to virtual reserves; liquidity relative to active liquidity; ranges relative to the current tick; pool metadata (fee, tick spacing, decimals). **[have]**
- **Splits:** chronological train/val/test by block, with purged horizons; 8 fitting pools and 2 held-out pools; scalers fit on training rows only. **[have]**
- **Supplementary data for §6:** 61M verified swaps from the six WETH/USDC pools on Arbitrum and Base (September 2025 to September 2026; 156/156 on-chain price checkpoints match) and Binance ETHUSDT 1-second prices. **[have]**
- **Table 1:** per-pool event counts, date ranges and split sizes. **[partial]** The numbers exist; the table needs writing.

## 4. Model

- **Event space:** sell, buy, mint, burn. The clock has a same-block continuation logit plus a geometric mixture over block gaps.
- **Marks:**
  - swap size: a rational-quadratic spline flow on log(base / virtual reserve), with buys truncated at exact capacity;
  - mint range: discretized logistic mixtures on the tick grid (lower bound and width);
  - mint amount: a flow on log(L / active L);
  - burn: range restricted to existing inventory, a full-burn Bernoulli, and a partial-fraction flow.
- **Encoder:** a Transformer over the last 64 events (12 features plus masks) with a metadata query token. The reference model is d=128, 4 layers and 0.95M parameters. **[have]**
- **Mechanics:** exact tick-crossing swap step and liquidity updates. Invalid sampled events become explicit absorbing failures, never silently fixed. **[have]**
- **Training:** next-event NLL, pool-balanced over 3 epochs (202M examples), AdamW. About 9 GPU-hours on one consumer GPU. **[have]**
- **Figure 1:** architecture and sampling loop, showing the learned versus deterministic split. **[run]** (drawing only)

## 5. Evaluation

### 5.1 Next-event likelihood

- **Have:** the reference model beats the smaller model and the pool-ID-embedding variant on all 10 pools. Seen macro NLL is 5.539 vs 5.609; held-out is 7.658 vs 7.717. Embeddings help seen pools slightly and hurt held-out ones. Per-head breakdown: size, mint lower/width/amount, burn. **[have]** `reports/pooled-full-20260930`
- **Missing baselines:**
  - an unconditional per-pool model (marks from empirical marginals, renewal clock);
  - a neural Hawkes or Transformer Hawkes model on the same event space;
  - ablations: no history (metadata only), and no mechanics-relative coordinates (raw sizes). **[run]**
- **Seeds:** three seeds of the reference model give seen macro NLL 5.529 ± 0.009 and held-out 7.640 ± 0.016. All three beat the 64×2 model on all 10 pools; the pool-ID embedding effect is within seed noise. **[have]** `paper/results/seed-summary.json`
- **Missing statistics:** bootstrap intervals over days. **[run]**
- **Table 2:** NLL by pool and head with baselines.

### 5.2 Rollout fidelity

- **Price distributions:** price CRPS and energy at 32 to 512 events and at clock horizons, against:
  - a random walk with trailing realized volatility;
  - a bootstrap of past returns;
  - optionally the direct non-recursive forecaster, which shows the "static cone" contrast.

  The current evidence uses 8 to 12 reused dates, so this needs hundreds of start points across pools and a fresh held-out period (October 2026 onward). **[partial → run]**
- **Asymmetric and state-dependent intervals:** conditional skew of simulated endpoints against realized outcomes, PIT histograms by regime, and examples where the predictive cone is lopsided ahead of liquidity cliffs. Per-path endpoints already exist in `pooled-rollout-train/…/rollouts`, `scratch-memory/…/rollouts` and the forecast-example exports. **[partial]**
- **Stylized facts table:** simulated versus real for return ACF, |return| ACF (volatility clustering), volatility against horizon, inter-arrival distribution, same-block clustering, side persistence, and the mint/burn mix. None of this exists for the pooled model yet; the legacy `audit.py` metrics can be ported. **[run]**
- **Figure 2:** forecast cones for several origins, with real paths overlaid. **[partial]** Data exists; the figure is new.
- **Figure 3:** stylized-facts panel. **[run]**

### 5.3 Liquidity behaviour

- **Mint and burn heads:** NLL and threshold Brier. The Brier fine-tune improves 10/10 pools by 2.3%. **[have]**
- **JIT-like sequences:**
  - The association is learned: in generated sequences the swap accompanying a mint is much larger than ordinary swaps (median 58,400 vs 3,599 USDC in USDC/WETH). **[have]**
  - Failure mode: generated swap size relative to deposited capital is too large (median 1.99% vs 0.11%). **[have]** `reports/jit-trends-20261005`
  - Failure mode: large mints stay unremoved after 64 events in 16–34% of generated cases against 0.28% historically. **[have]** `reports/jit-removal-followup-20261005`
- **Figure 4:** JIT scatter (deposit vs swap, generated vs historical) and a removal-survival curve. **[partial]** Data in JSON; figure new.

### 5.4 Likelihood is not simulation quality

- **Have:**
  - The larger model has better NLL, but 32-event energy and price CRPS get worse. **[have]**
  - Retained-history models trained from scratch: NLL improves (5.336 vs 5.346), but price CRPS at 128/256/512 events is 7–10% worse. **[have]** `reports/scratch-memory-20261011`
  - The extreme-path attribution (a liquidity cliff) is a case study of compounding errors. **[have]**
- **Keep only the from-scratch comparison.** The supervised-memory and rollout-context runs start from an LP-tuned parent, which would need explaining; mention them in one sentence or not at all.
- **Figure 5:** NLL vs rollout CRPS for the model variants. **[partial]**

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
- **Simulator test:** do model rollouts reproduce the conditional distribution of fees − LVR over the next interval, not just price? Score fees − LVR energy and coverage from rollouts against realized values at 5 min and 1 h. This is the bridge between §5 and §6. **[run]** The lp-outcome energy code is related, but it is measured against holding, not LVR.
- **Figure 6:** fees ÷ LVR across 16 pools and three chains with intervals.
- **Figure 7:** timing ceiling against decision interval.
- **Not claimed:** profitable trading. State plainly that no forecast-driven policy beat costs in our tests.

## 7. Limitations

- Single-pool histories: there is no cross-pool or CEX information in the model.
- Small-position overlay: our liquidity doesn't affect the flow.
- Ethereum-only training.
- JIT size mismatch, and rollout drift at long horizons.
- One training seed for the large runs, unless more are run.

## 8. Conclusion

Learning only the stochastic part, combined with exact mechanics, gives a faithful and inspectable AMM simulator. Next-event likelihood is the wrong target for long-horizon simulation. Multi-pool and CEX-conditioned simulation is the natural next step.

## Appendices

- A. Reconciliation algorithm and checks.
- B. Feature and target definitions; scaler protocol.
- C. Hyperparameters and compute.
- D. Mechanics edge cases: capacity truncation, uint128 limits, burn support, failure handling.
- E. LVR accounting derivation and synthetic validation.
- F. L2 data collection and verification.

---

## Work still to run, in priority order

1. **Stylized facts and asymmetry analysis** from rollouts, on new evaluation origins; CPU plus some GPU sampling.
2. **Baselines:** unconditional and Hawkes/TPP for NLL; random walk with trailing volatility and bootstrap for price CRPS.
3. **Larger, cleaner rollout evaluation:** hundreds of origins across pools, plus a fresh October 2026 held-out period (collect new logs, a few cents of RPC).
4. **Fees − LVR distribution test** from rollouts (§6 bridge).
5. **Seeds:** at least 2 more reference-model seeds (about 18 GPU-hours), or state single-seed plainly.
6. **Figures 1–7.**

## Repository status

The code lives in this repository: `dexnet/` (library) and `scripts/` (stages). The event model, data pipeline, direct baseline and economics scripts were extracted from the research codebase without behavioural changes. With this package:

- the reference event model reproduces its original per-pool test and held-out NLL exactly;
- the direct baseline reproduces its original per-pool objectives exactly;
- the fee/LVR, L2 and timing-ceiling outputs match the original runs exactly.

Decided:

- Title "DEX-Net"; author Clayton Barnes.
- Apache-2.0 for code; CC BY 4.0 for released data.
- Dataset released.
- Direct model included as a baseline.
- Two extra seeds of the reference model (260931, 260932) trained.

Completed experiments (results in `paper/results/`, run on the GPU box under `dexnet-paper-20261016`):

- **Rollout evaluation** (`calibration.json`, `stylized-facts.json`): 610 test anchors × 3 seeds × 32 paths at 32/128/512 events, against two history-only baselines. The model beats both baselines at 32 events in seen pools, ties the recent baseline at 128, and is worse at 512. Rare runaway paths, driven by too-heavy swap-size tails, dominate held-out long-horizon averages.
- **Conditional skew** (`skew.json`): predicted skew is not reliably directional, even in the most confident 20% or after large trends. The model carries recent momentum into its skew, while realised short-horizon moves revert slightly.
- **History ablation** (`history-ablation.json`): with no history, NLL is +1.08 nats in seen pools and +4.24 in held-out pools; with only the last 8 events, +0.165 and +0.32. Every pool and every component improves with history.
- **Path reweighting** (`reweight.json`): fitted on seen-pool validation rollouts only. It improves 512-event energy by 7.5% in seen pools and 51% in held-out pools, mainly by down-weighting runaway paths, but remains worse than the recent baseline at 512 events.

Still to add, as the paper experiments are run:

- a point-process (Hawkes) next-event baseline;
- the JIT analyses;
- figure scripts.
