"""AdamW with FP32 master weights and paired gradient accumulation."""

import math

import torch


class MasterAdam:
    """Update every parameter once after accumulating both probe conditions."""

    def __init__(
        self, model: torch.nn.Module, lr: float, *,
        fast_parameters=(), fast_multiplier: float = 1.0,
        fast_row_parameter=None, fast_row_indices=(),
    ):
        self.parameters = list(model.parameters())
        if not self.parameters or any(not p.requires_grad for p in self.parameters):
            raise ValueError("Full-parameter training requires every parameter to be trainable")
        if not math.isfinite(fast_multiplier) or fast_multiplier <= 0:
            raise ValueError("fast_multiplier must be finite and positive")
        fast_ids = {id(p) for p in fast_parameters}
        if not fast_ids.issubset({id(p) for p in self.parameters}):
            raise ValueError("Fast parameters must belong to the model")
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters]
        self.row_master = None
        if fast_row_parameter is not None:
            if fast_ids:
                raise ValueError("Choose fast whole parameters or fast rows, not both")
            matches = [i for i, p in enumerate(self.parameters) if p is fast_row_parameter]
            if not matches or fast_row_parameter.ndim != 2:
                raise ValueError("Fast row parameter must be a matrix belonging to the model")
            indices = list(fast_row_indices)
            if (
                not indices or len(set(indices)) != len(indices)
                or any(
                    type(i) is not int or not 0 <= i < fast_row_parameter.shape[0] for i in indices
                )
            ):
                raise ValueError("Fast row indices must be unique valid integers")
            self.row_parent = self.masters[matches[0]]
            self.row_indices = torch.tensor(indices, device=self.row_parent.device)
            self.row_master = torch.nn.Parameter(
                self.row_parent.detach().index_select(0, self.row_indices).clone()
            )
        elif fast_row_indices:
            raise ValueError("Fast row indices require a row parameter")
        groups = []
        for fast, scale in [(False, 1.0), (True, fast_multiplier)]:
            masters = [
                master for p, master in zip(self.parameters, self.masters, strict=True)
                if (id(p) in fast_ids) == fast
            ]
            if masters:
                groups.append({"params": masters, "lr": lr * scale, "lr_scale": scale})
        if self.row_master is not None:
            groups.append({
                "params": [self.row_master], "lr": lr * fast_multiplier, "lr_scale": fast_multiplier
            })
        self.optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0, foreach=False)
        self.accumulations = 0

    def accumulate(self) -> None:
        if self.accumulations >= 2:
            raise RuntimeError("Both conditions are already accumulated; call step()")
        if any(p.grad is None for p in self.parameters):
            raise RuntimeError("A parameter has no gradient; full-parameter training is required")
        for parameter, master in zip(self.parameters, self.masters, strict=True):
            gradient = parameter.grad.detach().float()
            if master.grad is None:
                master.grad = gradient.clone()
            else:
                master.grad.add_(gradient)
            parameter.grad = None
        self.accumulations += 1

    def step(self, lr: float) -> float:
        if self.accumulations != 2:
            raise RuntimeError("An update requires gradients from both conditions")
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group["lr_scale"]
        clipping_parameters = self.masters
        if self.row_master is not None:
            # Move these gradients, rather than duplicate or scale them. Their
            # Adam moments and learning rate live in the separate row parameter.
            self.row_master.grad = self.row_parent.grad.index_select(0, self.row_indices)
            self.row_parent.grad.index_fill_(0, self.row_indices, 0)
            clipping_parameters = [*self.masters, self.row_master]
        norm = torch.nn.utils.clip_grad_norm_(clipping_parameters, 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            if self.row_master is not None:
                self.row_parent.index_copy_(0, self.row_indices, self.row_master)
            for parameter, master in zip(self.parameters, self.masters, strict=True):
                parameter.copy_(master)
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulations = 0
        return float(norm)
