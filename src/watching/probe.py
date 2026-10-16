"""Read-only query-norm capture with an explicit compiled graph output."""

import torch


class ProbeModel(torch.nn.Module):
    """Expose a q_norm reference without replacing the activation."""

    def __init__(self, model: torch.nn.Module, layer: int):
        super().__init__()
        if not 0 <= layer < len(model.model.layers):
            raise ValueError(f"Probe layer {layer} is outside the model")
        self.model = model
        self.capture_enabled = False
        self.captured = []
        self.handle = model.model.layers[layer].self_attn.q_norm.register_forward_hook(self.record)

    def record(self, module, inputs, output) -> None:
        if self.capture_enabled:
            self.captured.append(output)
        # A hook must return None: returning a tensor replaces the module output.

    def forward(self, input_ids, attention_mask, position_ids, capture):
        self.capture_enabled = capture
        self.captured = []
        result = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=1,
        )
        return result.logits[:, -1, :], tuple(self.captured)
