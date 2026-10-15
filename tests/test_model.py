import math,unittest
import numpy as np
import torch
from dexnet.pool_mechanics import CapacityLedger,swap_token0,canonical_inventory
from dexnet.model import PooledEventModel,capacity_native,discrete_nll,apply_pooled_event
from dexnet.tick_mechanics import Q96,PoolState,sqrt_at_tick,ceil_div
from dexnet.training_data import pooled_collate


SCALERS=dict(targets=dict(mean=[-5.,0.,0.]+[0.]*5,std=[2.,1.,1.]+[1.]*5),raw_features=dict(mean=[0.,0.,0.,0.,0.,-3.]+[0.]*6,std=[1.]*12))


def initialized(spacing):
    ledger=CapacityLedger(spacing)
    common=dict(block=1,log_index=0,owner='a',amount0=0,amount1=0)
    ledger.apply(dict(common,event='Initialize',sqrt=Q96,tick=0))
    ledger.apply(dict(common,event='Mint',lower=-spacing*100,upper=spacing*100,amount=10**24))
    return ledger


def batch():
    rows=[]
    for kind in [0,1,2,3]:
        y=np.array([-5.,0.,0.,1.,-100.,200.,0.,12.]);mask=np.zeros(8,bool);mask[7]=True
        if kind<2:mask[0]=True
        else:mask[[4,5]]=True;mask[1 if kind==2 else 3]=True
        if kind==3:mask[6]=True
        rows.append(dict(x=np.zeros((64,12),np.float32),history_mask=np.ones((64,12),bool),history_kind=np.zeros(64,np.int8),kind=kind,
            targets=y,target_mask=mask,metadata=np.array([-7.,2.,1.,1.,1.],np.float32),anchor=0.,buy_log_cap=-1.,dt=1,lower=-100,width=200,
            grid_min=-88727,grid_max=88727,unit=10*np.log1p(.0001),lo=np.array([-100],np.int32) if kind==3 else [],wi=np.array([200],np.int32) if kind==3 else []))
    return pooled_collate(rows)


class PooledModelTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_token0_closed_form_and_fee_tiers(self):
        for fee in [100,500,3000]:
            ledger=initialized(10);state=PoolState(Q96,0,ledger.L);book=ledger.book();amount=10**18
            got,detail=swap_token0(state,book,amount,fee);net=amount*(1000000-fee)//1000000
            expected=ceil_div((state.liquidity<<96)*state.sqrt,(state.liquidity<<96)+net*state.sqrt)
            self.assertEqual(got.sqrt,expected);self.assertLess(detail['amount1'],0);self.assertEqual(state.sqrt,Q96)
            got,detail=swap_token0(state,book,-amount,fee)
            expected=ceil_div((state.liquidity<<96)*state.sqrt,(state.liquidity<<96)-amount*state.sqrt)
            self.assertEqual(got.sqrt,expected);self.assertGreater(detail['amount1'],0)

    def test_capacity_matches_direct_and_samples_execute_both_orientations(self):
        model=PooledEventModel(SCALERS).eval();b=batch();single={k:v[:1] for k,v in b.items() if k not in ['lower_candidates','lower_mask','width_candidates','width_mask']}
        with torch.no_grad():
            model.types[-1].weight.zero_();model.types[-1].bias.copy_(torch.tensor([-100.,100.,-100.,-100.]))
            model.mark_mean[0]=5.
        for spacing in [1,10,60]:
            ledger=initialized(spacing);state=PoolState(Q96,0,ledger.L)
            for base0 in [True,False]:
                meta=dict(tick_spacing=spacing,base_is_token0=base0,fee=500);inv=canonical_inventory(ledger,base0)
                self.assertEqual(ledger.capacity(base0),capacity_native(state,inv,meta))
                for seed in range(5):
                    torch.manual_seed(seed);event=model.sample_event(single,state,inv,meta)
                    self.assertEqual(event['kind'],1);self.assertLessEqual(-event['base_amount'],ledger.capacity(base0))
                    fresh=PoolState(Q96,0,ledger.L);book=ledger.book();apply_pooled_event(fresh,book,{lo:dict(up) for lo,up in inv.items()},event,meta)
                    self.assertGreater(fresh.sqrt,sqrt_at_tick(-spacing*100));self.assertLess(fresh.sqrt,sqrt_at_tick(spacing*100))

    def test_exact_capacity_across_bitmap_steps_and_gaps(self):
        from dexnet.pool_mechanics import swap_economic_base
        for base0 in [False,True]:
            ledger=initialized(1)
            for lo,hi,L in [(-700,-250,10**14),(250,700,10**14),(-400,400,10**12)]:
                ledger.apply(dict(event='Mint',owner='b',lower=lo,upper=hi,amount=L))
            state=PoolState(ledger.sqrt,ledger.tick,ledger.L);meta=dict(base_is_token0=base0,fee=100)
            cap=ledger.capacity(base0)
            final,_=swap_economic_base(state,ledger.book(),-cap,meta)
            self.assertEqual(final.liquidity,0)
            with self.assertRaises((ValueError,RuntimeError)):
                swap_economic_base(state,ledger.book(),-(cap+1),meta)


    def test_buy_likelihood_preserves_sub_float32_capacity_margin(self):
        m=PooledEventModel(SCALERS).eval();b=batch()
        b['buy_log_cap'][1]=np.nextafter(float(b['targets'][1,0]),math.inf)
        self.assertEqual(float(b['buy_log_cap'][1].float()),float(b['targets'][1,0].float()))
        loss,_=m.losses(b);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))


    def test_supported_density_integrates(self):
        m=PooledEventModel(SCALERS).double();m.mark_mean[0]=0;m.mark_std[0]=1
        x=torch.linspace(-10,.5-1e-8,4001,dtype=torch.float64);raw=m.flow.identity_bias().double()[None].repeat(len(x),1)
        density=(-m.amount_nll(raw,x,0,torch.full_like(x,.5))).exp()
        self.assertAlmostEqual(float(torch.trapezoid(density,x)),1.,places=5)

    def test_discrete_density_normalization_different_grids(self):
        for spacing in [1,10,60]:
            grids=torch.arange(-5,6).float();raw=torch.tensor([[0.]*4+[0.]*4+[1.]*4]).expand(len(grids),-1)
            anchor=torch.full_like(grids,.25);unit=torch.full_like(grids,spacing*math.log1p(.0001))
            nll=discrete_nll(raw,grids,anchor,unit,torch.full_like(grids,-5),torch.full_like(grids,5))
            self.assertAlmostEqual(float((-nll).exp().sum()),1.,places=5)

    def test_masked_targets_no_effect_and_all_heads_finite_gradients(self):
        m=PooledEventModel(SCALERS).eval();b=batch();loss,parts=m.losses(b);self.assertTrue(torch.isfinite(loss))
        altered=dict(b,targets=b['targets'].clone());altered['targets'][0,1]=1e30;altered['targets'][2,0]=1e30
        other,_=m.losses(altered);torch.testing.assert_close(loss,other,rtol=0,atol=0)
        loss.backward();self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        for head in ['clock','types','swap_size','mint_amount','burn_full','lower','width']:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in getattr(m,head).parameters() if p.grad is not None),0)

    def test_partial_burn_and_narrow_pool_rounding_allowance(self):
        m=PooledEventModel(SCALERS).eval();b=batch();b['targets'][3,3]=0;b['targets'][3,2]=.5;b['target_mask'][3,2]=True
        loss,_=m.losses(b);loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.burn_fraction.parameters()),0)
        ledger=CapacityLedger(1);ledger.apply(dict(event='Initialize',sqrt=Q96,tick=0))
        ledger.apply(dict(event='Mint',block=1,log_index=0,owner='a',lower=-1,upper=1,amount=10**12))
        from dexnet.tick_mechanics import delta0
        geometric=delta0(Q96,sqrt_at_tick(1),ledger.L,False)
        self.assertGreater(ledger.capacity(True),geometric-100)


if __name__=='__main__':unittest.main()
