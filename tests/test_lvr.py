"""Synthetic checks for the fee/LVR accounting: exact single-swap terms and sigma^2/8."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dexnet.lvr import swap_terms
from scripts.fee_lvr import analyse_pool

FIRST, FEE, VOL = 1_000_000, .0005, .65


def fixture(root, n=60_000, noise=1., seed=1):
    """USDC token0 / WETH token1 pool: CEX random walk, arbitrage to the fee band, noise swaps."""
    rng = np.random.default_rng(seed); times = 1_600_000_000+12*np.arange(n)
    cex = 3000*np.exp(np.cumsum(rng.normal(0, VOL/np.sqrt(365*7200), n)))
    sq = lambda p: str(int(np.sqrt(1e12/p)*2**96))
    pool, rows = 3000., []
    for i in range(n):
        moves = []
        if cex[i] > pool*(1+FEE): moves.append(cex[i]/(1+FEE))
        elif cex[i] < pool/(1+FEE): moves.append(cex[i]*(1+FEE))
        for _ in range(rng.poisson(1.)):
            moves.append((moves[-1] if moves else pool)*np.exp(rng.normal(0, noise*VOL/np.sqrt(365*7200))))
        for log, new in enumerate(moves):
            rows.append((FIRST+i, log, 'Swap', sq(pool), sq(new))); pool = new
    d = root/'pools'/'TEST-500'; (d/'decoded').mkdir(parents=True); (root/'headers').mkdir()
    b, l, e, s0, s1 = zip(*rows)
    pq.write_table(pa.table(dict(block=np.array(b), log_index=np.array(l), event=list(e), sqrt_before=list(s0), sqrt_after=list(s1))), d/'decoded'/'00000.parquet')
    (d/'metadata.json').write_text(json.dumps(dict(address='0xtest', fee=500, decimals0=6, decimals1=18, base='WETH', quote='USDC', base_is_token0=False)))
    (d/'exposure.json').write_text(json.dumps([dict(split='train', first_block=FIRST, last_block=FIRST+n-1, first_timestamp=int(times[0]), last_timestamp=int(times[-1]))]))
    (root/'headers'/'COMPLETE.json').write_text(json.dumps(dict(first_block=FIRST)))
    np.save(root/'headers'/'timestamp.npy', times)
    return times.astype(float), cex


class FeeLVRTests(unittest.TestCase):
    def test_single_swap_terms(self):
        a, b = np.array([50., 50.]), np.array([51., 49.])  # sqrt prices: price up, price down
        fees, loss = swap_terms(a, b, FEE, np.zeros(2), b**2)
        rate = FEE/(1-FEE)
        self.assertAlmostEqual(fees[0], (51-50)*rate)                # quote input
        self.assertAlmostEqual(fees[1], (1/49-1/50)*rate*49**2)       # base input at post price
        np.testing.assert_allclose(loss, (b-a)**2/a)                  # discrete hedge cost

    def test_cex_lvr_matches_sigma_squared_over_eight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cex = fixture(root)
            _, table = analyse_pool(root, 'TEST-500', cex=cex)
            days = len(table['hour'])/24
            expected = VOL**2/365/8
            for key in ('cex+0s', 'pool+50b'):
                self.assertAlmostEqual(table[f'lvr:{key}'].sum()/days/expected, 1, delta=.2, msg=key)
            # Fees are identical whichever markout is used for the base leg, to first order.
            self.assertAlmostEqual(table['fee:cex+0s'].sum()/table['fee:pool+0b'].sum(), 1, delta=.01)


if __name__ == '__main__':
    unittest.main()
