import unittest,math
import numpy as np
import torch
from copy import deepcopy
from dexnet.diagnostics import proper_scores,feedback,endpoint,ENDPOINT_SCALES
from dexnet.scores import energy_surrogate
from dexnet.model import PooledEventModel,apply_pooled_event
from dexnet.pool_mechanics import canonical_inventory
from dexnet.scores import mint_brier,mint_exceedance
from dexnet.tick_mechanics import PoolState,Q96
from tests.test_model import initialized,SCALERS


class DiagnosticTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_energy_matches_previously_tested_training_score(self):
        x=np.random.default_rng(11).normal(size=(16,4));y=np.array([1.,2.,3.,4.]);score,crps=proper_scores(x,y)
        _,old=energy_surrogate(torch.tensor(x/ENDPOINT_SCALES)[None],torch.tensor(y/ENDPOINT_SCALES)[None],torch.zeros(1,16))
        self.assertAlmostEqual(score,float(old),places=10)
        for i in range(4):
            _,old=energy_surrogate(torch.tensor(x[:,i])[None,:,None],torch.tensor([[y[i]]]),torch.zeros(1,16))
            self.assertAlmostEqual(crps[i],float(old),places=10)
    def test_feedback_masks_and_token_orientation(self):
        for base0 in [False,True]:
            meta=dict(base_is_token0=base0,tick_spacing=10,fee=500);ledger=initialized(10)
            state=PoolState(Q96,0,ledger.L);inv=canonical_inventory(ledger,base0);book=ledger.book();initial=deepcopy(state)
            event=dict(kind=0,dt=3,base_amount=10**18)
            apply_pooled_event(state,book,inv,event,meta);x,mask=feedback(initial,state,event,meta,np.zeros(12),np.ones(12))
            self.assertAlmostEqual(x[0],math.log1p(36),places=6);self.assertAlmostEqual(x[2],math.log(10**18/initial.liquidity),places=6)
            self.assertTrue(mask[2]);self.assertFalse(mask[3]);self.assertLess(x[1],0)
            self.assertAlmostEqual(endpoint(state,initial,3,10**18,meta)[0],x[1]*1e4,places=5)
    def test_mint_brier_backpropagates_with_pooled_flow(self):
        m=PooledEventModel(SCALERS);h=torch.randn(7,64);raw=m.mint_amount(h).double();y=torch.linspace(-2,8,7,dtype=torch.float64)
        score=mint_brier(m,raw,y).mean();score.backward();p=mint_exceedance(m,raw)
        self.assertTrue(torch.isfinite(score));self.assertTrue(((p>=0)&(p<=1)).all());self.assertTrue((p[:,1:]<=p[:,:-1]).all())
        self.assertGreater(sum(float(v.grad.abs().sum()) for v in m.mint_amount.parameters()),0)


if __name__=='__main__':unittest.main()
