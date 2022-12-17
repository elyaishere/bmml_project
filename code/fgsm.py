import torch


class FGSM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, eps=0.001, **kwargs):
        assert rho >= 0.0

        defaults = dict(rho=rho, eps=eps, **kwargs)
        super(FGSM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False, save_state=False):
        for group in self.param_groups:
            scale = group["rho"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None: continue
                if save_state: self.state[p]["old_p"] = p.data.clone()
                e_w = scale * torch.sign(p.grad)
                p.add_(e_w)
                p = torch.max(torch.min(p, self.state[p]["old_p"] + eps), self.state[p]["old_p"] - eps)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
