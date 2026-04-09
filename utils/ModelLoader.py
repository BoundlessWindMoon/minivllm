import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

class ModelLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def default_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def inject_data(self, model: nn.Module):
        packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
        for file in glob(os.path.join(self.data_path, "*.safetensors")):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    for k in packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            break
                    else:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", self.default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
        return model