"""Bounded, permutation-equivariant weights for frozen path ensembles (post-hoc reweighting).

Each simulated path gets one weight from its own summary features; weights keep a
uniform floor and bounded logits, so no path is dropped and no outcome is invented.
The score is the proper energy score of the weighted (finite) distribution.
"""
import torch
from torch import nn


class PathWeighter(nn.Module):
    def __init__(self, features, context, conditional=True, rank=4, floor=.1, bound=2.):
        super().__init__();self.floor=floor;self.bound=bound
        self.bias=nn.Parameter(torch.zeros(features))
        self.context=nn.Sequential(nn.Linear(context,rank),nn.Tanh(),nn.Linear(rank,features,bias=False)) if conditional else None
        if self.context is not None:nn.init.zeros_(self.context[-1].weight)
    def forward(self,x,context):
        coefficient=self.bias[None]
        if self.context is not None:coefficient=coefficient+self.context(context)
        logits=self.bound*torch.tanh((x*coefficient[:,None]).sum(-1)/x.shape[-1]**.5)
        return (1-self.floor)*logits.softmax(-1)+self.floor/x.shape[1]


def energy(weights,to_truth,pairwise):
    """Proper score of the issued finite weighted distribution (includes diagonal).

    This is not the fair iid-ensemble estimator: learned unequal weights define
    the issued distribution. Training and all comparisons use this same score.
    """
    return (weights*to_truth).sum(-1)-.5*torch.einsum('bi,bij,bj->b',weights,pairwise,weights)


def diagnostics(w):
    return dict(effective_paths=1/w.square().sum(-1),max_weight=w.max(-1).values,
        kl_to_uniform=(w*(w*w.shape[-1]).log()).sum(-1))


def distances(x,y):
    return torch.linalg.vector_norm(x-y[:,None],dim=-1),torch.cdist(x,x,compute_mode='donot_use_mm_for_euclid_dist')
