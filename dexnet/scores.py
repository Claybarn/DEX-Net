"""Proper scores for sampled paths and mint-size threshold calibration."""
import math
import torch
from .flow import rq_spline


def energy_surrogate(samples,target,log_scores):
    """Unbiased U-statistic energy score and its stochastic gradient surrogate.

    samples B x P x D; each log score sums all discrete choices on that path.
    Continuous samples remain reparameterized, including recurrent state updates.
    The score-function coefficient for path i includes every pair involving i.
    A leave-one-path-out baseline is independent of path i conditional on context.
    """
    p=samples.shape[1]
    if p<3:raise ValueError('Need at least three paths for independent baseline')
    a=torch.linalg.vector_norm(samples-target[:,None,:],dim=-1)
    distances=torch.linalg.vector_norm(samples[:,:,None,:]-samples[:,None,:,:],dim=-1)
    pair_sum=distances.sum((1,2))
    score=a.mean(1)-pair_sum/(2*p*(p-1))
    # Full-score coefficient conditional on this path, omitting independent terms.
    reward=a/p-distances.sum(-1)/(p*(p-1))
    others_a=(a.sum(1,keepdim=True)-a)/(p-1)
    others_pairs=(pair_sum[:,None]-2*distances.sum(-1))/((p-1)*(p-2))
    baseline=(others_a-others_pairs)/p
    sf=((reward-baseline).detach()*log_scores).sum(1)
    surrogate=(score+sf-sf.detach()).mean()
    return surrogate,score.detach().mean()


THRESHOLDS = (1., 10., 100., 1000.)


def mint_exceedance(model, raw):
    """Differentiable P(amount / pre-event active liquidity > threshold)."""
    shift, scale, widths, heights, derivatives = model.flow.split(raw)
    probabilities = []
    for threshold in THRESHOLDS:
        value = raw.new_full(raw.shape[:-1], math.log(threshold))
        z = (value-model.mark_mean[1])/model.mark_std[1]
        u, _ = rq_spline((z-shift)*torch.exp(-scale), widths, heights,
                         derivatives, bound=model.flow.bound)
        probabilities.append(.5*torch.erfc(u/math.sqrt(2)))
    return torch.stack(probabilities, -1)


def mint_brier(model, raw, log_ratio):
    probabilities = mint_exceedance(model, raw)
    thresholds = log_ratio.new_tensor(THRESHOLDS).log()
    observed = (log_ratio[:, None] > thresholds).to(probabilities.dtype)
    return (probabilities-observed).square().mean(-1)
