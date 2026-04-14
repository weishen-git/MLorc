import math
import torch
from torch.optim.optimizer import Optimizer
from rsvd import randomized_svd



class MLorc_Lion(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.95, 0.98), weight_decay=0.05, rank=4, p=0, torchapi=False):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        self.rank=rank
        super().__init__(params, defaults)



    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.dim() != 2:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                    
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values

                    state["m_u"] = torch.zeros((p.data.shape[0], self.rank), dtype=p.data.dtype, device=p.data.device)
                    state["m_v"] = torch.zeros((self.rank, p.data.shape[1]), dtype=p.data.dtype, device=p.data.device)
                    state["m_s"] = torch.zeros((self.rank), dtype=p.data.dtype, device=p.data.device)

                m_u, m_v, m_s= state["m_u"], state["m_v"], state["m_s"]
                beta1, beta2 = group["betas"]

                m=m_u @ torch.diag(m_s) @ m_v
                update=(beta1 * m + (1-beta1) * grad).sign_()

                state["step"] += 1
                step_size = group["lr"]
                p.data.add_(update, alpha=-step_size)

                m_=beta2 * m + (1-beta2) * grad
                m_u, m_s, m_v = randomized_svd(m_, self.rank)
                state["m_u"], state["m_v"], state["m_s"] = m_u, m_v, m_s
                
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss