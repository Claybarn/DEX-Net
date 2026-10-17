"""Conditional 1-D normalizing flow for swap sizes.

A rational-quadratic spline (Durkan et al., 2019, "Neural Spline Flows")
with linear tails, preceded by a conditional affine map. The transform is
monotone with an analytic inverse, so log-likelihood (forward) and sampling
(inverse) are both exact and cheap. Used as an alternative to the Gaussian
mixture on z = (log size - loc) / scale.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

MIN_BIN_WIDTH = 1e-3
MIN_BIN_HEIGHT = 1e-3
MIN_DERIVATIVE = 1e-3
# softplus(x) + MIN_DERIVATIVE == 1  ->  identity-slope initialization
DERIV_IDENTITY_BIAS = math.log(math.exp(1.0 - MIN_DERIVATIVE) - 1.0)


def n_spline_params(n_bins: int) -> int:
    """widths + heights + interior derivatives."""
    return 3 * n_bins - 1


def _searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    bin_locations = bin_locations.clone()
    bin_locations[..., -1] += 1e-6
    return (torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1).clamp(min=0, max=bin_locations.shape[-1] - 2)


def _rq_spline_core(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool,
    bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spline on [-bound, bound]^2; ``unnormalized_derivatives`` has n_bins+1 entries."""
    n_bins = unnormalized_widths.shape[-1]
    left = bottom = -bound
    right = top = bound

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = MIN_BIN_WIDTH + (1.0 - MIN_BIN_WIDTH * n_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = MIN_DERIVATIVE + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = MIN_BIN_HEIGHT + (1.0 - MIN_BIN_HEIGHT * n_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = _searchsorted(cumheights if inverse else cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)
        discriminant = (b.pow(2) - 4.0 * a * c).clamp(min=0.0)
        root = (2.0 * c) / (-b - torch.sqrt(discriminant))
        root = root.clamp(0.0, 1.0)
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1.0 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2.0 * input_delta * theta_one_minus_theta
            + input_derivatives * (1.0 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2.0 * torch.log(denominator)
        return outputs, -logabsdet

    theta = ((inputs - input_cumwidths) / input_bin_widths).clamp(0.0, 1.0)
    theta_one_minus_theta = theta * (1.0 - theta)
    numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta) * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator
    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2.0 * input_delta * theta_one_minus_theta
        + input_derivatives * (1.0 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2.0 * torch.log(denominator)
    return outputs, logabsdet


def rq_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    bound: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unconstrained RQ spline: identity (linear tails) outside [-bound, bound].

    ``unnormalized_derivatives`` has n_bins-1 interior entries; the boundary
    derivatives are fixed to 1 so the map is continuous with the tails.
    Returns (outputs, log|det|) for the given direction.
    """
    inside = (inputs >= -bound) & (inputs <= bound)
    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)
    if not bool(inside.any()):
        return outputs, logabsdet
    derivs = F.pad(unnormalized_derivatives, (1, 1), value=DERIV_IDENTITY_BIAS)
    out_in, lad_in = _rq_spline_core(
        inputs[inside],
        unnormalized_widths[inside],
        unnormalized_heights[inside],
        derivs[inside],
        inverse=inverse,
        bound=bound,
    )
    outputs = outputs.masked_scatter(inside, out_in)
    logabsdet = logabsdet.masked_scatter(inside, lad_in)
    return outputs, logabsdet


def rq_spline_static(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    bound: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """rq_spline without data-dependent shapes (no host synchronisation).

    Every element goes through the spline core with its input clamped into
    [-bound, bound]; elements outside keep the identity tail via torch.where, so
    their spline evaluation is finite and receives zero gradient. Values equal
    rq_spline's.
    """
    inside = (inputs >= -bound) & (inputs <= bound)
    derivs = F.pad(unnormalized_derivatives, (1, 1), value=DERIV_IDENTITY_BIAS)
    out_in, lad_in = _rq_spline_core(inputs.clamp(-bound, bound), unnormalized_widths, unnormalized_heights, derivs,
                                     inverse=inverse, bound=bound)
    return torch.where(inside, out_in, inputs), torch.where(inside, lad_in, torch.zeros_like(lad_in))


class ConditionalSizeFlow:
    """Stateless helper: maps raw head outputs to log-prob / samples.

    Parameter layout per row: [shift, log_scale, widths(K), heights(K),
    derivatives(K-1)]. Forward direction (data -> base):
        u1 = (z - shift) * exp(-log_scale);  u = spline(u1);  u ~ N(0, 1).
    """

    def __init__(self, n_bins: int, bound: float = 6.0) -> None:
        self.n_bins = n_bins
        self.bound = bound
        self.n_params = 2 + n_spline_params(n_bins)

    def split(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.n_bins
        shift = raw[..., 0]
        log_scale = raw[..., 1].clamp(-4.0, 4.0)
        widths = raw[..., 2 : 2 + k]
        heights = raw[..., 2 + k : 2 + 2 * k]
        derivs = raw[..., 2 + 2 * k : 2 + 3 * k - 1]
        return shift, log_scale, widths, heights, derivs

    def log_prob(self, raw: torch.Tensor, z: torch.Tensor, static: bool = False) -> torch.Tensor:
        """log density of z (normalized log size) under the conditional flow."""
        shift, log_scale, w, h, d = self.split(raw)
        u1 = (z - shift) * torch.exp(-log_scale)
        u, lad = (rq_spline_static if static else rq_spline)(u1, w, h, d, inverse=False, bound=self.bound)
        base = -0.5 * u.pow(2) - 0.5 * math.log(2.0 * math.pi)
        return base + lad - log_scale

    def inverse(self, raw: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Map base samples u to z (normalized log size)."""
        shift, log_scale, w, h, d = self.split(raw)
        u1, _ = rq_spline(u, w, h, d, inverse=True, bound=self.bound)
        return u1 * torch.exp(log_scale) + shift

    def identity_bias(self) -> torch.Tensor:
        """Head bias that makes the flow the identity at init (z ~ N(0,1))."""
        k = self.n_bins
        bias = torch.zeros(self.n_params)
        bias[2 + 2 * k :] = DERIV_IDENTITY_BIAS
        return bias
