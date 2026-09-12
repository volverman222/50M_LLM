import torch
from torch.optim import Optimizer


@torch.no_grad()
def zeropower_via_newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Aproxima la polarización ortogonal de una matriz 2D."""
    X = G.float()
    X /= X.norm() + 1e-7

    # Coeficientes Newton–Schulz de quinto orden.
    a, b, c = 3.4445, -4.7750, 2.0315

    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    return X.T if transposed else X


class Muon(Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95,
                 weight_decay=0.0, ns_steps=5):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon solo admite parámetros matriciales 2D")

                grad = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                # Regularización desacoplada, como AdamW.
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)

                update = zeropower_via_newton_schulz5(
                    buf, steps=group["ns_steps"]
                ).to(dtype=p.dtype)

                # Escala recomendada para matrices no cuadradas.
                update.mul_((p.size(0) / p.size(1)) ** 0.5)
                p.add_(update, alpha=-lr)

        return loss