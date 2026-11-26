from torch import nn


class LandscapeEncoder(nn.Module):
    """Encode standardized landscape features into decoder context."""

    def __init__(self, feature_dim, d_model, hidden_dim):
        super().__init__()
        if feature_dim <= 0 or d_model <= 0 or hidden_dim <= 0:
            raise ValueError("Landscape encoder dimensions must be positive.")
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, features):
        if features is None:
            raise ValueError("Continual ALDes mode requires problem features.")
        if features.ndim == 3:
            features = features.mean(dim=1)
        if features.ndim != 2:
            raise ValueError(
                "Problem features must have shape (batch, features) or "
                "(batch, samples, features)."
            )
        return self.network(features)
