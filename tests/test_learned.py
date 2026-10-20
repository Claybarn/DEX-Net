import math,unittest
from types import SimpleNamespace
import numpy as np
import torch
from tests.test_model import SCALERS,batch
from dexnet.events import event_features
from dexnet.learned import LearnedTransitionModel,learned_features,advance,RET,LIQ,LP_RATIO
from dexnet.model import native_range
from dexnet.tick_mechanics import Q96,PoolState,sqrt_at_tick
from scripts.rollout_eval import learned_trajectory

MEAN=np.array(SCALERS['raw_features']['mean']);STD=np.array(SCALERS['raw_features']['std'])


def learned_batch():
    b=batch();n=len(b['kind']);x=torch.zeros((n,12));m=torch.zeros((n,12),dtype=torch.bool)
    # sell, buy, mint, burn: returns only on swaps, liquidity changes on the buy and the mint.
    x[:,RET]=torch.tensor([-.002,0.,0.,0.]);x[:,LIQ]=torch.tensor([0.,-.05,.01,0.]);x[2:,LP_RATIO]=torch.tensor([-4.,-6.])
    m[:,[RET,LIQ]]=True;m[2:,LP_RATIO]=True
    b['next_x']=((x.double()-torch.tensor(MEAN))/torch.tensor(STD)).float();b['next_mask']=m
    return b


class LearnedTransitionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(3)

    def test_loss_is_finite_and_every_new_head_learns(self):
        model=LearnedTransitionModel(SCALERS,d_model=32,layers=1);loss,parts=model.losses_static(learned_batch());loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(parts['return'][1]),2);self.assertEqual(int(parts['liquidity'][1]),4);self.assertEqual(int(parts['burn_amount'][1]),1)
        for name in ['ret_zero','ret_size','liq_zero','liq_size','burn_amount']:
            grad=sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters())
            self.assertGreater(grad,0,name)

    def test_history_features_match_the_exact_builder(self):
        for base0 in [True,False]:
            meta=dict(tick_spacing=10,base_is_token0=base0);sign=1 if base0 else -1;tick=1234
            for lower,width in [(sign*123-3,5),(sign*123,1),(sign*123-1,1),(sign*123+2,4)]:
                event=dict(kind=2,dt=3,z=-4.,lower=lower,width=width,anchor=sign*tick/10,ret=0.,liquidity_change=.01)
                f,mask=learned_features(event,meta,MEAN,STD)
                lo,hi=native_range(lower,lower+width,meta);L=10**20
                r=dict(event='Mint',block=3,timestamp=36,amount=round(L*math.exp(-4.)),amount0=0,amount1=0,lower=lo,upper=hi)
                post=SimpleNamespace(L=round(L*math.exp(.01)),sqrt=sqrt_at_tick(tick),tick=tick)
                exact,exact_mask,*_=event_features(r,(L,sqrt_at_tick(tick),tick),post,meta,(0,0),0)
                np.testing.assert_array_equal(mask,exact_mask)
                np.testing.assert_allclose(f,np.where(exact_mask,(exact-MEAN)/STD,0),rtol=1e-6,atol=1e-6,err_msg=f'{base0} {lower} {width}')

    def test_state_tracks_price_tick_and_reserve(self):
        state=dict(log_price=0.,log_liquidity=50.,tick=100.,log_reserve=20.)
        advance(state,dict(ret=.01,liquidity_change=-.2),dict(base_is_token0=False))
        self.assertAlmostEqual(state['tick'],100-.01/math.log1p(.0001));self.assertAlmostEqual(state['log_reserve'],20-.2-.005)
        with self.assertRaises(ValueError):advance(state,dict(ret=0.,liquidity_change=-60.),dict(base_is_token0=True))

    def test_rollout_runs_and_reports_endpoints(self):
        model=LearnedTransitionModel(SCALERS,d_model=32,layers=1).eval();b=learned_batch()
        single={k:b[k][:1] for k in ['x','history_mask','history_kind','metadata']}
        initial=PoolState(Q96,0,10**20);meta=dict(tick_spacing=10,base_is_token0=True,metadata_features=[0.]*5)
        out=learned_trajectory(model,single,initial,meta,MEAN,STD,(1,8),7,'cpu')
        if out['failure'] is None:
            self.assertEqual(set(out['values']),{'1','8'});self.assertEqual(len(out['trace']['kind']),8)
            self.assertEqual(out['values']['8'][2],math.log1p(sum(out['trace']['dt'])))
        else:self.assertIn('step',out['failure'])


if __name__=='__main__':unittest.main()
