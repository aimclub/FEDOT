import torch


def torch_pdist(X: torch.Tensor, metric: str = 'euclidean', p: float = 3):
    if metric == 'euclidean':
        diff = X.unsqueeze(0) - X.unsqueeze(1)
        dist = torch.sqrt((diff * diff).sum(dim=-1))

    elif metric == 'cityblock':
        diff = (X.unsqueeze(0) - X.unsqueeze(1)).abs()
        dist = diff.sum(dim=-1)

    elif metric == 'chebyshev':
        diff = (X.unsqueeze(0) - X.unsqueeze(1)).abs()
        dist = diff.max(dim=-1).values

    elif metric == 'minkowski':
        diff = (X.unsqueeze(0) - X.unsqueeze(1)).abs() ** p
        dist = (diff.sum(dim=-1)) ** (1.0 / p)

    elif metric == 'cosine':
        Xn = X / (X.norm(p=2, dim=1, keepdim=True))
        sim = Xn @ Xn.T
        dist = 1.0 - sim

    elif metric == 'correlation':
        Xm = X - X.mean(dim=1, keepdim=True)
        Xn = Xm / (Xm.norm(p=2, dim=1, keepdim=True))
        sim = Xn @ Xn.T
        dist = 1.0 - sim

    elif metric == 'canberra':
        A = X.unsqueeze(0)
        B = X.unsqueeze(1)
        num = (A - B).abs()
        denom = A.abs() + B.abs()
        zero_mask = denom == 0
        denom = torch.where(zero_mask, torch.ones_like(denom), denom)
        frac = num / denom
        frac = torch.where(zero_mask, torch.zeros_like(frac), frac)
        dist = frac.sum(dim=-1)

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return dist
