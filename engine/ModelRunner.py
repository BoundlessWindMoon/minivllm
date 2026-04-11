import torch

class ModelRunner:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    @torch.inference_mode()  
    def compute_logits(self, input_ids: torch.Tensor, position: torch.Tensor, is_prefill: bool) -> torch.Tensor:

        hidden_states = self.model(
            input_ids=input_ids.to(self.device),
            positions=position.to(self.device),
        )
        
        logits = self.model.compute_logits(hidden_states)
        return logits

    def post_process(self, input_ids: torch.Tensor, position: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        new_input_ids = torch.cat([input_ids, next_token], dim=-1)
        new_position = position[:, -1:] + 1
        new_position_ids = torch.cat([position, new_position], dim=-1)
        return new_input_ids, new_position_ids
