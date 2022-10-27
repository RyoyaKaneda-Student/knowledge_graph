"""Hyperbolic operations utilModules functions."""

import torch
from torch import tanh, sinh, cosh, atanh

# MIN_NORM = 1e-15
# BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


# ################# HYP OPS ########################
@torch.jit.script
def project(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Project points to Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with projected hyperbolic points.
    """
    MIN_NORM = torch.tensor(1e-15, requires_grad=False, dtype=torch.float32)

    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = 4e-3
    maxnorm = (1. - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


@torch.jit.script
def expmap0(u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Args:
        u: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with tangent points.
    """
    MIN_NORM = torch.tensor(1e-15, requires_grad=False, dtype=torch.float32)

    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    tmp = sqrt_c * u_norm
    gamma_1 = tanh(tmp) * u / tmp
    return project(gamma_1, c)


@torch.jit.script
def logmap0(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Logarithmic map taken at the origin of the Poincare ball with curvature c.

    Args:
        y: torch.Tensor of size B x d with tangent points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with hyperbolic points.
    """
    MIN_NORM = torch.tensor(1e-15, requires_grad=False, dtype=torch.float32)

    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    tmp = sqrt_c * y_norm
    return y / tmp * atanh(tmp)



@torch.jit.script
def mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Mobius addition of points in the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        y: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        Tensor of shape B x d representing the element-wise Mobius addition of x and y.
    """
    MIN_NORM = torch.tensor(1e-15, requires_grad=False, dtype=torch.float32)

    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


# ################# HYP DISTANCES ########################

def hyp_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor):
    """Hyperbolic distance on the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size 1 with absolute hyperbolic curvature
        eval_mode: bool of eval_mode

    Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    sqrt_c = c ** 0.5
    _xy = -mobius_add(x, y, c)
    print(_xy)
    norm_xy = torch.norm(_xy, p=2, dim=-1, keepdim=True)
    print(norm_xy)

    return (2. / sqrt_c) * atanh(norm_xy * sqrt_c)


def hyp_distance_multi_c(x: torch.Tensor, v: torch.Tensor, c: torch.Tensor, eval_mode=False):
    """Hyperbolic distance on Poincare balls with varying curvatures c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        v: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size B x d with absolute hyperbolic curvatures
        eval_mode: bool of eval_mode

    Return: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    MIN_NORM = torch.tensor(1e-15, requires_grad=False, dtype=torch.float32)

    sqrt_c = c ** 0.5
    if eval_mode:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1)
        xv = x @ v.transpose(0, 1) / vnorm
    else:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True)
        xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True)
    gamma = tanh(sqrt_c * vnorm) / sqrt_c
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = atanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


def test():
    print()
    import utils.math.hyperbolic_old as old_
    x = torch.tensor([0.0, 0.1, 0.2])
    y = torch.tensor([0.3, 0.4, 0.5])
    c = torch.tensor(1)
    assert torch.equal(old_.artanh(x), atanh(x))
    assert torch.equal(old_.expmap0(x, c), expmap0(x, c))
    assert torch.equal(old_.logmap0(x, c), logmap0(x, c))
    assert torch.equal(old_.mobius_add(x, y, c), mobius_add(x, y, c))
    x = torch.stack((x, x))
    y = torch.stack((y, y))
    x = expmap0(x, c)
    y = expmap0(y, c)
    tmp1 = old_.hyp_distance(x, y, c)
    print(tmp1)
    tmp2 = hyp_distance(x, y, c)
    print(tmp2)
    assert torch.equal(tmp1, tmp2)
