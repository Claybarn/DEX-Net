import itertools
import numpy as np
import torch
from dexnet.rollout import clock_log_prob,draw_flow
from dexnet.scores import energy_surrogate
from dexnet.model import PooledEventModel,discrete_nll,bounds
from dexnet.base_model import interval_log_prob


def model():
    return PooledEventModel({'targets':{'mean':[0.,0.,0.],'std':[1.,1.,1.]},'raw_features':{'mean':[0.]*12}},d_model=16,layers=1).double().eval()


def test_clock_normalized_score_and_gradient():
    raw=torch.tensor([[.3, .2,-.3,.5, .3,-.2,.1]],dtype=torch.double,requires_grad=True)
    dt=torch.arange(100,dtype=torch.double)
    probs=clock_log_prob(raw.expand(100,-1),dt).exp()
    assert torch.allclose(probs.sum(),torch.ones((),dtype=torch.double),atol=1e-12)
    grad=torch.autograd.grad(probs.sum(),raw)[0]
    assert grad.abs().max()<1e-12


def test_two_step_hard_recurrent_energy_gradient_exact_enumeration():
    # First choice changes the second choice and endpoint. No pathwise derivatives.
    theta=torch.tensor(.2,dtype=torch.double,requires_grad=True);paths=3;expectation=theta*0;surrogate=theta*0
    for bits in itertools.product([0.,1.],repeat=2*paths):
        endpoints=[];scores=[];weight=theta.new_ones(())
        for i in range(paths):
            first,second=bits[2*i:2*i+2];p=theta.sigmoid();q=(theta+2*first).sigmoid()
            prob=(p if first else 1-p)*(q if second else 1-q)
            weight=weight*prob; scores.append(prob.log()); endpoints.append([first+3*second])
        x=theta.new_tensor(endpoints)[None];y=theta.new_tensor([[1.5]])
        loss,score=energy_surrogate(x,y,torch.stack(scores)[None])
        expectation=expectation+weight*score;surrogate=surrogate+weight.detach()*loss
    assert torch.allclose(torch.autograd.grad(expectation,theta,retain_graph=True)[0],torch.autograd.grad(surrogate,theta)[0],atol=1e-12)


def test_flow_latents_detached_but_score_has_gradient_and_buy_support():
    m=model();raw=torch.zeros((8,m.flow.n_params),dtype=torch.double,requires_grad=True)
    # Flow exposes params via the actual head if interface changes.
    for cap in [None,torch.full((8,),.2,dtype=torch.double)]:
        z,score=draw_flow(m,raw,0,cap)
        assert not z.requires_grad
        if cap is not None:assert (z<cap).all()
        g=torch.autograd.grad(score,raw)[0]
        assert torch.isfinite(g).all() and g.abs().sum()>0


def test_supported_grid_probabilities_and_gradients():
    raw=torch.zeros((1,12),dtype=torch.double,requires_grad=True)
    candidates=torch.tensor([[-7.,0.,2.,19.]],dtype=torch.double);mask=torch.ones_like(candidates,dtype=torch.bool)
    anchor=torch.tensor([1.],dtype=torch.double);unit=torch.tensor([.001],dtype=torch.double)
    ps=[]
    for value in candidates[0]:ps.append((-discrete_nll(raw,value[None],anchor,unit,anchor,anchor,candidates=candidates,mask=mask)).exp())
    total=torch.stack(ps).sum();assert abs(float(total)-1)<1e-12
    assert torch.autograd.grad(total,raw)[0].abs().max()<1e-12

def test_exact_simulator_rollout_gradients_and_failed_paths_retained():
    from unittest.mock import patch
    from tests.test_model import initialized,SCALERS,batch
    from dexnet.pool_mechanics import canonical_inventory
    from dexnet.tick_mechanics import PoolState,Q96
    from dexnet.rollout import rollout_loss
    m=PooledEventModel(SCALERS,d_model=16,layers=1).eval();ledger=initialized(10)
    b={k:v[:1] for k,v in batch().items() if k in ['x','history_kind','history_mask','metadata']}
    meta=dict(base_is_token0=True,tick_spacing=10,fee=500)
    r=dict(state=vars(PoolState(Q96,0,ledger.L)),meta=meta,inventory=canonical_inventory(ledger,True),actual={'8':[0.,0.,1.,.01]})
    torch.manual_seed(721)
    loss,info=rollout_loss(m,b,r,ledger.book(),np.zeros(12),np.ones(12),[10.,1.,1.,.01],paths=4)
    loss.backward();assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert sum(float(p.grad.abs().sum()) for p in m.clock.parameters())>0
    # A failure is an observed absorbing outcome, not a missing ensemble member.
    def fail(model,h,*args):return dict(kind=2,dt=0,error='test failure'),model.clock(h).sum()
    with patch('dexnet.rollout.sampled_event',side_effect=fail):
        loss,info=rollout_loss(m,b,r,ledger.book(),np.zeros(12),np.ones(12),[10.,1.,1.,.01],paths=4)
    assert len(info['endpoints'])==4 and all(x[-1]==10 for x in info['endpoints'])
    assert all(x['step']==1 for x in info['failures'])
    assert torch.isfinite(loss)


import unittest
class RolloutGradientTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    test_simulator = staticmethod(test_exact_simulator_rollout_gradients_and_failed_paths_retained)
    test_clock = staticmethod(test_clock_normalized_score_and_gradient)
    test_recurrent_energy = staticmethod(test_two_step_hard_recurrent_energy_gradient_exact_enumeration)
    test_flow = staticmethod(test_flow_latents_detached_but_score_has_gradient_and_buy_support)
    test_grid = staticmethod(test_supported_grid_probabilities_and_gradients)

if __name__=='__main__':unittest.main()
