# DEX-Net

A mechanics-grounded generative event model of Uniswap v3 order flow.

DEX-Net learns only the stochastic parts of a pool's activity: when the next event arrives, whether it is a buy, sell, mint or burn, how large it is relative to available depth, and which tick range liquidity is added to or removed from. Prices and liquidity are then computed exactly from pool state with Uniswap v3 tick mechanics. A sampled path is therefore always a valid pool history; a sampled event that cannot be executed ends the path and is reported, never silently repaired.

The repository contains everything behind the accompanying paper:

- **Data:** collection and exact reconstruction of ten Ethereum v3 pools (May 2021 – September 2026, 29.4M events, active liquidity reconciled at every swap).
- **Model:** a pooled Transformer event model (0.95M parameters in the reference configuration) and a direct multi-horizon baseline.
- **Evaluation:** anchor reconstruction, multi-step rollouts and proper scores.
- **Economics:** fee income versus loss-versus-rebalancing (LVR) per swap, against on-chain or Binance prices; Arbitrum and Base collection; and a perfect-foresight bound on the value of LP timing.

## Install

Python 3.10+.

```bash
uv venv && uv pip install -e .
```

The only dependencies are `numpy`, `torch`, `pyarrow` and `pandas`. Training uses one GPU (CUDA or ROCm); everything else runs on CPU.

```bash
python -m unittest discover -s tests -t .
```

## Pipeline

All commands run from the repository root. Each stage writes into its own output folder, refuses to overwrite a finished stage, and can be resumed after interruption.

### 1. Data

Collection needs an Ethereum endpoint that serves historical `eth_getLogs` and archive `eth_call`. Endpoints come from environment variables and are never written to disk.

```bash
export DEXNET_RPC_URLS=https://your-endpoint      # comma-separated for rotation
python -m scripts.collect_logs    --out data/raw
python -m scripts.collect_headers --out data/headers     # exact block times from the public AWS dataset
python -m scripts.prepare_data    --raw data/raw --headers data/headers --out data/prepared
python -m scripts.prepare_context --data data/prepared --out data/context
python -m scripts.audit_mechanics --data data/prepared --out data/mechanics-audit
```

`prepare_data` decodes every Initialize/Swap/Mint/Burn log and replays it through an exact-integer liquidity ledger. It requires the reconstructed active liquidity to equal the pool's reported liquidity at every swap, and it checks the block-header hash chain. It then builds chronological train/validation/test splits (with purged horizons), fits scalers on training rows of the eight training pools only, and keeps two pools (USDC/USDT 0.05%, WBTC/WETH 0.3%) entirely held out.

### 2. Train the event model

```bash
python -m scripts.prepare_training --data data/prepared --context data/context --out runs/ref --seed 260930
python -m scripts.train --out runs/ref --smoke     # optional 128-batch GPU check
python -m scripts.train --out runs/ref
```

The reference model is d_model 128, 4 layers and 4 heads, with a 64-event context. It trains for 3 pool-balanced epochs (202M examples) with AdamW, a learning-rate peak of 1e-4, a 1,000-update warmup and cosine decay. Training takes about 9 GPU-hours on one Radeon RX 7900 XTX. Validation selects `best.pt`, which is then scored on the test and held-out pools. The paper reports seeds 260930, 260931 and 260932.

### 3. Evaluate rollouts

```bash
python -m scripts.prepare_anchors --data data/prepared --out eval/anchors-test --split test --anchors 64 --horizons 32,128,512
python -m scripts.rollout_eval --anchors eval/anchors-test --out eval/rollouts-test \
    --model ref=runs/ref/training/best.pt --paths 32 --workers 8
```

Anchors are spaced uniformly in event index, never selected by outcome, and their windows don't overlap. Each anchor's pool state and per-range inventory are rebuilt exactly from the raw log, and every real feature row inside the window is re-derived and checked against the dataset. The rollout evaluation scores each model's paths on price return, liquidity change, elapsed blocks and volume. It reports the unbiased ensemble energy score, CRPS, bias and 90% coverage, and saves every path for further analysis.

### 4. Direct baseline

```bash
python -m scripts.prepare_direct_labels --data data/prepared --out runs/direct/labels
python -m scripts.train_direct --data data/prepared --labels runs/direct/labels --out runs/direct/seed1 --init-seed 88011 --train-seed 61001
```

### 5. LP economics

```bash
python -m scripts.fetch_binance --symbol ETHUSDT --months 2025-09:2026-09 --out data/cex
python -m scripts.collect_protocol_fees --data data/prepared --out data/protocol
python -m scripts.fee_lvr --data data/prepared --out results/fee-lvr --protocol-dir data/protocol \
    --cex data/cex/ethusdt_1s.parquet --cex-pools USDC-WETH-500,USDC-WETH-3000,USDT-WETH-500,USDT-WETH-3000
DEXNET_RPC_URL_BASE=... python -m scripts.collect_l2_swaps --chain base --out data/l2
DEXNET_RPC_URL_BASE=... python -m scripts.verify_l2_swaps  --chain base --out data/l2
python -m scripts.l2_fee_lvr --l2 data/l2 --cex data/cex/ethusdt_1s.parquet --mainnet results/fee-lvr --out results/l2
python -m scripts.timing_ceiling --data data/prepared --protocol-dir data/protocol --l2 data/l2 \
    --cex data/cex/ethusdt_1s.parquet --out results/ceiling.json
python -m scripts.hourly_gate --ledgers results/fee-lvr --out results/gate
```

For a liquidity position that stays in range, `fee_lvr` computes each swap's exact fee income (taken on the input leg, net of the protocol fee where its history is provided) and its adverse-selection loss at a reference price. Summed per block, that loss is the cost of rebalancing a delta hedge, i.e. LVR. The loss is measured at markouts against the pool's own later price or against Binance. `timing_ceiling` finds the best possible in/out schedule with perfect foresight at decision intervals from one block to one day, with entry costs. No forecaster can beat that bound.

`collect_protocol_fees` writes one `<pool>-protocol.json` per pool with the pool's `SetFeeProtocol` history, reconciled against on-chain state at both ends. Without these files, fees are reported gross of the protocol fee.

## Layout

```
dexnet/        library: v3 tick mechanics, event dataset, model, sampler, scores, LVR accounting
scripts/       command-line stages (python -m scripts.<name>)
tests/         unit tests, including exact-mechanics and synthetic LVR checks
paper/         paper source and outline
```

## Data release

The decoded and reconciled per-pool event data, the Arbitrum/Base swap collections, and the reference checkpoints will be released alongside the paper. Released data is under CC BY 4.0. All of it derives from public blockchain records; Binance prices are not redistributed, and `scripts.fetch_binance` downloads them from Binance's public archive.

## License

Code: [Apache License 2.0](LICENSE). Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

```bibtex
@misc{barnes2026dexnet,
  title  = {DEX-Net: A Mechanics-Grounded Generative Model of Uniswap v3 Order Flow},
  author = {Barnes, Clayton},
  year   = {2026},
  note   = {arXiv preprint}
}
```
