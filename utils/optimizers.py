import torch
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by You, Gitman, and Ginsburg.
    https://arxiv.org/abs/1708.03888
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        weight_decay=0.0001,
        trust_coefficient=0.001,
        eps=1e-8,
        exclude_bias_and_norm=True,
    ):
        """
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            lr (float): learning rate
            momentum (float, optional): momentum factor (default: 0.9)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0001)
            trust_coefficient (float, optional): trust coefficient for computing adaptive lr (default: 0.001)
            eps (float, optional): epsilon for numerical stability (default: 1e-8)
            exclude_bias_and_norm (bool): exclude bias and norm layers from LARS adaptation (default: True)
        """
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )
        self.exclude_bias_and_norm = exclude_bias_and_norm
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            trust_coefficient = group["trust_coefficient"]
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Apply weight decay before computing adaptive LR
                if weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=weight_decay)

                # Skip bias and normalization layers for LARS adaptation
                if self.exclude_bias_and_norm:
                    param_norm = torch.norm(p.data)
                    if param_norm == 0.0 or len(p.shape) == 1:  # bias or BatchNorm
                        p.grad.data.add_(p.data, alpha=weight_decay)
                        continue

                param_norm = torch.norm(p.data)
                update_norm = torch.norm(p.grad.data)
                
                # Compute adaptive learning rate
                if param_norm != 0 and update_norm != 0:
                    # Lars coefficient
                    lars_coef = trust_coefficient * param_norm / (update_norm + param_norm * weight_decay + eps)
                    # LARS scaling for this parameter
                    local_lr = lr * lars_coef
                else:
                    local_lr = lr

                # SGD with momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(p.grad).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(p.grad)
                    
                    p.add_(buf, alpha=-local_lr)
                else:
                    p.add_(p.grad, alpha=-local_lr)

        return loss 