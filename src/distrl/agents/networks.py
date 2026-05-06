import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPTrunk(nn.Module):
    """
    Common MLP-based feature extractor for vector observations.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list[int] = [128, 128], 
        activation: type[nn.Module] = nn.ReLU
    ) -> None:
        super(MLPTrunk, self).__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(activation())
            curr_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = curr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QHead(nn.Module):
    """
    Head for standard Q-Learning. 
    Outputs a single Q-value estimate for each discrete action.
    """
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super(QHead, self).__init__()
        self.fc = nn.Linear(input_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class QuantileHead(nn.Module):
    """
    Head for Distributional RL (Quantile Regression).
    Outputs a set of quantiles representing the return distribution for each action.
    """
    def __init__(self, input_dim: int, action_dim: int, num_quantiles: int) -> None:
        super(QuantileHead, self).__init__()
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        
        # num sortides = actions x num_quantiles    
        self.fc = nn.Linear(input_dim, action_dim * num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: [batch_size, action_dim * num_quantiles]
        quantiles = self.fc(x)
        # Reshape to: [batch_size, action_dim, num_quantiles]
        # Use -1 for batch dimension to avoid graph breaks in torch.compile
        return quantiles.view(-1, self.action_dim, self.num_quantiles)

class UnifiedQNet(nn.Module):
    """
    A unified network that combines a trunk with a specific head.
    """
    def __init__(self, trunk: nn.Module, head: nn.Module) -> None:
        super(UnifiedQNet, self).__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        return self.head(features)
