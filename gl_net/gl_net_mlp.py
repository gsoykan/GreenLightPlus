from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        # nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class GLNetMLP(nn.Module):
    def __init__(self,
                 input_a_dims: Optional[int] = 293,
                 input_p_dims: Optional[int] = 254,
                 input_d_dims: int = 10,
                 input_u_dims: int = 11,
                 input_x_dims: int = 28,
                 output_dim: int = 28,
                 input_layers_out_dim: int = 64,
                 fc_out_dim: int = 64,
                 arc_variation: int = 1,
                 use_final_tanh: bool = False,
                 ):
        """
        Optional dims values result in reduced number of parameters, good for experimentation
        Args:
            input_dims (Dict[str, int]): A dictionary with keys 'a', 'd', 'p', 'u', 'x' and their corresponding input dimensions.
            output_dim (int): The dimension of the output vector.
        """
        super(GLNetMLP, self).__init__()
        self.input_a_dims = input_a_dims
        self.input_d_dims = input_d_dims
        self.input_p_dims = input_p_dims
        self.input_u_dims = input_u_dims
        self.input_x_dims = input_x_dims
        self.output_dim = output_dim
        self.arc_variation = arc_variation
        self.use_final_tanh = use_final_tanh

        self.input_layers_out_dim = input_layers_out_dim
        self.fc_out_dim = fc_out_dim

        self.input_dims = {
            "a": self.input_a_dims,
            "d": self.input_d_dims,
            "p": self.input_p_dims,
            "u": self.input_u_dims,
            "x": self.input_x_dims
        }

        if self.input_a_dims is None:
            del self.input_dims["a"]

        if self.input_p_dims is None:
            del self.input_dims["p"]

        match arc_variation:
            case 1:
                input_layers_intermediate_dim = 2 * self.input_layers_out_dim
                self.input_layers = nn.ModuleDict({
                    key: nn.Sequential(
                        nn.Linear(dim, input_layers_intermediate_dim),
                        # nn.BatchNorm1d(input_layers_intermediate_dim),
                        # nn.LayerNorm(input_layers_intermediate_dim),
                        nn.ReLU(),
                        nn.Linear(input_layers_intermediate_dim, self.input_layers_out_dim),
                        # nn.BatchNorm1d(self.input_layers_out_dim),
                        # nn.LayerNorm(self.input_layers_out_dim),
                        nn.ReLU()
                    ) for key, dim in self.input_dims.items()
                })

                total_input_dim = sum([self.input_layers_out_dim for _ in self.input_dims])
                self.fc = nn.Sequential(
                    nn.Linear(total_input_dim, self.fc_out_dim),
                    # nn.BatchNorm1d(self.fc_out_dim),
                    # nn.LayerNorm(self.fc_out_dim),
                    nn.ReLU(),
                    nn.Linear(self.fc_out_dim, output_dim)
                )
            case 2:
                input_layers_intermediate_dim = len(self.input_dims) * 2 * self.input_layers_out_dim
                total_input_dim = sum(self.input_dims.values())
                self.input_layers = nn.ModuleDict({
                    key: nn.Identity() for key, dim in self.input_dims.items()
                })
                self.fc = nn.Sequential(
                    nn.Linear(total_input_dim, input_layers_intermediate_dim),
                    nn.ReLU(),
                    nn.Linear(input_layers_intermediate_dim, input_layers_intermediate_dim),
                    nn.ReLU(),
                    nn.Linear(input_layers_intermediate_dim, self.fc_out_dim),
                    nn.ReLU(),
                    nn.Linear(self.fc_out_dim, output_dim)
                )
            case _:
                raise ValueError(f"Invalid arc variation: {arc_variation}")

        if self.use_final_tanh:
            self.fc.append(nn.Tanh())

        self.apply(init_weights)

    def forward(self,
                input_vector_dict: Dict[str, Tensor]) -> Tensor:
        processed_inputs = [self.input_layers[k](v) for k, v in input_vector_dict.items() if k in self.input_dims]
        combined_input = torch.cat(processed_inputs, dim=1)  # [B, total_input_dim]
        output = self.fc(combined_input)  # [B, output_dim]
        return output


if __name__ == '__main__':
    gl_net = GLNetMLP()
    io_batched_instance_path = '/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLightPlus/gl_net/io_batched_instance.pt'
    io_batched_instance = torch.load(io_batched_instance_path)
    io_batched_input, io_batched_output = io_batched_instance

    criterion = nn.MSELoss()
    with torch.no_grad():
        output = gl_net(io_batched_input)
        loss = criterion(output, io_batched_output)

    print(output, loss.item())
