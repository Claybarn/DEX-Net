import unittest
from unittest.mock import patch
import numpy as np
import torch
from dexnet.trace import event_token,transition,sample_traces,SequenceDiscriminator,NAMES
from dexnet.model import PooledEventModel,apply_pooled_event
from dexnet.rollout import rollout_loss
from dexnet.pool_mechanics import canonical_inventory
from dexnet.tick_mechanics import PoolState,Q96
from tests.test_model import SCALERS,batch,initialized


class RawTraceTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_observed_and_simulated_transition_encoding_identical(self):
        ledger=initialized(10);before=PoolState(Q96,0,ledger.L);after=PoolState(Q96,0,ledger.L)
        meta=dict(base_is_token0=True,tick_spacing=10,fee=500);event=dict(kind=0,dt=2,base_amount=10**16)
        detail=apply_pooled_event(after,ledger.book(),canonical_inventory(ledger,True),event,meta)
        generated=transition(event,before,after,detail,meta)
        observed=dict(kind=0,dt=2,amount0=str(detail['amount0']),amount1=str(detail['amount1']),amount=0,
            sqrt_before=str(before.sqrt),sqrt_after=str(after.sqrt),liquidity_before=str(before.liquidity),liquidity_after=str(after.liquidity),tick_before=before.tick,tick_after=after.tick)
        np.testing.assert_array_equal(event_token(generated,before,meta),event_token(observed,before,meta))
        self.assertEqual(len(event_token(observed,before,meta)),len(NAMES))

    def test_full_trace_capture_matches_existing_mechanics(self):
        ledger=initialized(10);m=PooledEventModel(SCALERS,d_model=16,layers=1).eval()
        b={k:v[:1] for k,v in batch().items() if k in ['x','history_kind','history_mask','metadata']}
        r=dict(state=vars(PoolState(Q96,0,ledger.L)),meta=dict(base_is_token0=True,tick_spacing=10,fee=500),inventory=canonical_inventory(ledger,True),actual={'8':[0,0,1,.01]})
        def swap(model,h,*args):return dict(kind=0,dt=1,base_amount=10**16),h.sum()*0
        with patch('dexnet.trace.sampled_event',side_effect=swap):
            x,info=sample_traces(m,b,r,ledger.book(),np.zeros(12),np.ones(12),paths=4,steps=8)
        with patch('dexnet.rollout.sampled_event',side_effect=swap),torch.no_grad():
            _,old=rollout_loss(m,b,r,ledger.book(),np.zeros(12),np.ones(12),[10,1,1,.01],paths=4,steps=8)
        np.testing.assert_allclose(info['endpoints'],np.array(old['endpoints'])[:,:4],rtol=0,atol=0)
        self.assertEqual(x.shape,(4,8,len(NAMES)))
        for trace in info['events']:
            for a,b in zip(trace,trace[1:]):self.assertEqual(a['sqrt_after'],b['sqrt_before'])

    def test_sequence_model_uses_order_and_all_input_gradients_finite(self):
        torch.manual_seed(11);m=SequenceDiscriminator();h=torch.randn(3,64,len(NAMES));f=torch.randn(3,32,len(NAMES),requires_grad=True);meta=torch.randn(3,7)
        y=m(h,f,meta);self.assertFalse(torch.allclose(y,m(h,f.flip(1),meta)))
        y.sum().backward();self.assertTrue(torch.isfinite(f.grad).all());self.assertGreater(float(f.grad.abs().sum()),0)
        torch.testing.assert_close(m(h,f,meta,'history'),m(h,f*100,meta,'history'))

    def test_failures_retained_as_full_length_audited_sequences(self):
        ledger=initialized(10);m=PooledEventModel(SCALERS,d_model=16,layers=1).eval()
        b={k:v[:1] for k,v in batch().items() if k in ['x','history_kind','history_mask','metadata']}
        r=dict(state=vars(PoolState(Q96,0,ledger.L)),meta=dict(base_is_token0=True,tick_spacing=10,fee=500),inventory=canonical_inventory(ledger,True))
        def fail(model,h,*args):return dict(kind=2,dt=0,error='test'),h.sum()*0
        with patch('dexnet.trace.sampled_event',side_effect=fail):x,info=sample_traces(m,b,r,ledger.book(),np.zeros(12),np.ones(12),paths=3,steps=4)
        self.assertEqual(x.shape,(3,4,len(NAMES)));self.assertTrue(np.isfinite(x).all());self.assertTrue(all(info['failures']))


if __name__=='__main__':unittest.main()
