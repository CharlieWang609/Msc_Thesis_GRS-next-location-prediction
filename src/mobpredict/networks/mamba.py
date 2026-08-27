import torch
import torch.nn as nn
from torch import Tensor
from mamba_ssm import Mamba

from mobpredict.networks.embed import AllEmbedding
from mobpredict.networks.fc import FullyConnected

class MambaEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(MambaEncoder, self).__init__()
        
        self.d_input = config.base_emb_size
        self.Embedding = AllEmbedding(self.d_input, config)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            Mamba(
                d_model=self.d_input,     # Model dimension d_model
                d_state=config.d_state,    # SSM state expansion factor
                d_conv=config.d_conv,      # Local convolution width
                expand=config.expand_factor, # Block expansion factor
            ) for _ in range (config.num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.d_input)
        
        # Fully connected output layer
        self.fc = FullyConnected(self.d_input, config, if_residual_layer=True)
        
        self._init_weights()
        
    def forward(self, src, context_dict, device) -> Tensor:
        # Get embeddings
        emb = self.Embedding(src, context_dict)  # [seq_len, batch, d_model]
        seq_len = context_dict["len"]
        
        # Mamba expects [batch, sequence, feature].
        x = emb.transpose(0, 1)
        
        # Apply Mamba layers
        for layer in self.layers:
            x = layer(x)  # Mamba2 maintains the input shape
        
        # Reshape back to [seq_len, batch, d_model]
        x = x.transpose(0, 1)
        
        # Final normalization
        x = self.norm(x)
        
        # Get the last timestep output for each sequence
        out = x.gather(
            0,
            seq_len.view([1, -1, 1]).expand([1, x.shape[1], x.shape[-1]]) - 1,
        ).squeeze(0)
        
        # Apply final fully connected layer
        return self.fc(out)
    
    def _init_weights(self):
        """Initialize network parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
