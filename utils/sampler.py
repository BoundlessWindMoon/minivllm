import torch
from typing import Optional
import torch.nn.functional as F


class Sampler(object):
    def __init__(
        self,
        sample_method: Optional[str] = "greedy",
        temperature: Optional[float] = 1.0,
        top_k: int = 0,
        top_p: float = 0,
    ):
        self.sample_method = sample_method
        self.temperature = temperature
        self.topk = top_k
        self.topp = top_p

    def sample(self, logits):
        logits = logits[:, -1, :]

        # Fast path: compact logits from kernel argmax (shape [batch, 1])
        if logits.shape[-1] == 1:
            return logits.long()

        if self.sample_method == "greedy" or self.temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        if self.temperature != 1.0:
            logits = logits / self.temperature

        if self.topk > 0:
            top_k_val = torch.topk(logits, self.topk, dim=-1)[0][:, -1:]
            logits = logits.masked_fill(logits < top_k_val, float('-inf'))

        if 0.0 < self.topp < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs - sorted_probs > self.topp
            sorted_logits[sorted_indices_to_remove] = float('-inf')

            logits = sorted_logits.scatter(
                dim=-1, index=sorted_indices, src=sorted_logits
            )
        probs = F.softmax(logits, dim=-1)
        sampled_indices = torch.multinomial(probs, num_samples=1)
        return sampled_indices
