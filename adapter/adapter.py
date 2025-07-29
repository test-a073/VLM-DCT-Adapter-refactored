import torch
import torch.nn as nn
import math
import torch.nn.functional as F

#best v4
class DCTAdapter(nn.Module):
    def __init__(self, in_features=768, num_components=24, tau=1.0):
        super().__init__()
        self.tau = tau
        self.adapter_gate_logits = nn.Parameter(torch.randn(in_features))  # One per DCT dim
        self.adapter_down = nn.Linear(in_features, num_components, bias=False) # 18 best
        self.adapter_up = nn.Linear(num_components, in_features, bias=False) # 18 best

    def gumbel_softmax_mask(self, logits):
        gumbel_noise = -torch.empty_like(logits).exponential_().log()
        y = logits + gumbel_noise
        return F.softmax(y / self.tau, dim=-1)  # soft but approximates hard gate

    def dct1(self, x):
        N = x.size(-1)
        
        # Check and clean input
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = torch.clamp(x, -1e4, 1e4)

        # Safely create DCT matrix
        dct_mat = self.create_dct_matrix(N, x.device, torch.float64).to(dtype=x.dtype)
        
        # Sanity check
        if torch.isnan(dct_mat).any():
            raise ValueError("DCT matrix contains NaNs")
        
        return x @ dct_mat.T


    def idct1(self, x):
        N = x.size(-1)

        # Check and clean the input
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = torch.clamp(x, -1e4, 1e4)

        # Create the DCT matrix in higher precision
        dct_mat = self.create_dct_matrix(N, x.device, torch.float64).to(dtype=x.dtype)

        # Sanity check the matrix
        if torch.isnan(dct_mat).any() or torch.isinf(dct_mat).any():
            raise ValueError("DCT matrix has NaN or Inf values!")

        result = x @ dct_mat

        # Optional: clip the result to avoid propagation
        result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)

        return result


    def create_dct_matrix(self, N, device=None, dtype=torch.float32):
        n = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)
        k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
        dct = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
        dct[0] *= 1 / math.sqrt(2)
        dct *= math.sqrt(2 / N)
        return dct

    def forward(self, hidden_states):
        dct = self.dct1(hidden_states)  # [B, T, C]

        # gate_mask = self.gumbel_softmax_mask(self.adapter_gate_logits)  # [C]
        # gated_dct = dct * gate_mask  # broadcasted over B and T

        
        # print("dct\n", dct)
        z = dct.reshape(-1, dct.shape[-1])  # [B*T, C]
        # print("z\n",z)
        adapter_down_out = self.adapter_down(z)
        # print("Adapter down out\n",adapter_down_out)
        relu_out = F.leaky_relu(adapter_down_out)
        # print("Relu out\n", relu_out)
        z_pert = self.adapter_up(relu_out)
        
        out = z_pert.view_as(dct)
        # print("out\n", out)
        idct = self.idct1(out)
        
        return hidden_states + idct  # residual connection
        
