import torch
import torch.optim

from .utils import warmup, exp_avg_sq_, beta_debias, update_param_, StatefulOptimizer


def linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
    if step < warmup:
        a = step / float(warmup)
        return (1.0-a) * alpha_start + a * alpha_end
    return alpha_end


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):

    def f(beta, eps=1e-8):
        return math.log(0.5)/math.log(beta+eps)-1

    def f_inv(t):
        return math.pow(0.5, 1/(t+1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0-a) * f(beta_start) + a * f(beta_end))
    return beta_end


class ForeachLaProp(StatefulOptimizer):

    def __init__(self, params, lr=0.0025, betas=(0.9, 0.999, 0.999), alpha=2.0, eps=1e-8, beta3_warmup=None,
                 alpha_warmup=None, weight_decay=0, warmup_steps=1, foreach: bool = True):
        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps, k=0, warmup_steps=warmup_steps, train_mode=True, weight_sum=0.0,
                        lr_max=-1.0, beta3_warmup=beta3_warmup, alpha_warmup=alpha_warmup, weight_decay=weight_decay)
        super().__init__(params, defaults, foreach)

    def _step(self, group):
        eps = group['eps']
        decay = group['weight_decay']
        k = group['k']
        beta1, beta2, beta3_final = group["betas"]
        beta3_warmup = group["beta3_warmup"]
        alpha_final = group["alpha"]
        alpha_warmup = group["alpha_warmup"]

        if not group['train_mode']:
            raise Exception("Not in train mode!")

        active_p = [p for p in group['params'] if p.grad is not None]

        if not active_p:
            return

        for p in active_p:
            if 'exp_avg_slow' not in self.state_(p):
                if beta1 != 0.0: # save memory in case beta1 is 0.0
                    self.state_(p)['exp_avg_fast'] = torch.zeros_like(p.data, dtype=torch.float32)
                else: 
                    self.state_(p)['exp_avg_fast'] = None
                self.state_(p)['exp_avg_slow'] = torch.zeros_like(p.data, dtype=torch.float32)
                self.state_(p)['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

        y, grad, exp_avg_sq, exp_avg_slow, exp_avg_fast = zip(
            *[(p.data, p.grad.float(), self.state_(p)['exp_avg_sq'], self.state_(p)['exp_avg_slow'], self.state_(p)['exp_avg_fast']) for p in active_p])

        # Compute the effective alpha and beta3 in case warmup is used 
        if alpha_warmup is not None:
            alpha = linear_warmup_scheduler(state["step"], alpha_end=alpha_final, alpha_start=0, warmup=alpha_warmup)
        else:
            alpha = alpha_final
        
        if beta3_warmup is not None:
            beta3 = linear_hl_warmup_scheduler(state["step"], beta_end=beta3_final, beta_start=beta1, warmup=beta3_warmup)
        else:
            beta3 = beta3_final

        # Decay the first and second moment running average coefficient
        denom = exp_avg_sq_(exp_avg_sq, grad, beta_debias(beta2, k + 1), eps)
        if beta1 != 0.0:
            beta1 = beta_debias(beta1, k + 1)
            torch._foreach_mul_(exp_avg_fast, beta1)
            torch._foreach_addcdiv_(exp_avg_fast, grad, denom, 1 - beta1)
        else:
            exp_avg_fast = grad
        beta3 = beta_debias(beta3, k + 1)
        torch._foreach_mul_(exp_avg_slow, beta3)
        torch._foreach_addcdiv_(exp_avg_slow, grad, denom, 1 - beta3)
        del grad

        # Normalize grad in-place for memory efficiency
        lr = -warmup(group['lr'], k + 1, group['warmup_steps'])
        update_param_(y, exp_avg_slow, exp_avg_fast, lr, decay)

        group['k'] = k + 1
